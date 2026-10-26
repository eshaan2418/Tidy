"""Synthetic human phone-video demonstrations (sim fallback for B-003).

A free-floating hand proxy (mocap wrist + forearm, thumb tip, index tip with coloured caps) performs the
chore at human speed with human-like kinematics, and a handheld "phone" camera on the human side records
RGB-D at 30 fps.  Ground-truth annotations (keypoints, events, phases, object poses) are stored for the M3
perception metrics.  None of the GT is visible to the learning pipeline, which only reads frames.

Human motion model (assumptions, documented in DECISIONS D-013):
  * minimum-jerk point-to-point segments, peak speeds 0.4-0.9 m/s, varied per demo;
  * the hand approaches from the human side (+x) at 25-60 deg from vertical; pinch grasps with
    the thumb-index axis across the object's narrow dimension;
  * the hand carries in an arc and drops/places at the chore's goal (same goal semantics as the program);
  * small tremor (1-2 mm), random pick order unless the chore requires one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from common.geometry import min_jerk, wrap_half_pi
from eval.chores import ChoreSpec, goal_position
from eval.placements import LIGHTING, PlacementCard
from sim.env import TidyEnv
from sim.expert import grasp_height, grasp_yaw, jaw_half_extent

FPS = 30
PHONE_W, PHONE_H = 320, 240
PHONE_FOVY = 60.0


@dataclass
class PhoneCamera:
    pos: np.ndarray
    look_at: np.ndarray
    up_roll: float = 0.0

    def xyaxes(self) -> np.ndarray:
        f = self.look_at - self.pos
        f = f / np.linalg.norm(f)
        right = np.cross(f, np.array([0, 0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        c, s = np.cos(self.up_roll), np.sin(self.up_roll)
        r2 = c * right + s * up
        u2 = -s * right + c * up
        return np.concatenate([r2, u2])

    def intrinsics(self) -> np.ndarray:
        fy = 0.5 * PHONE_H / np.tan(np.deg2rad(PHONE_FOVY) / 2)
        return np.array([[fy, 0, PHONE_W / 2], [0, fy, PHONE_H / 2], [0, 0, 1.0]])

    def world_to_cam(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns R, t with x_cam = R @ x_world + t, OpenCV convention (x right, y down, z forward)."""
        ax = self.xyaxes()
        right, up = ax[:3], ax[3:]
        fwd = -np.cross(right, up)
        R = np.stack([right, -up, fwd])
        return R, -R @ self.pos


@dataclass
class HandPose:
    wrist: np.ndarray
    thumb: np.ndarray
    index: np.ndarray


@dataclass
class DemoGT:
    t: np.ndarray                              # (T,) seconds
    hand: dict[str, np.ndarray]                # wrist/thumb/index (T,3)
    objects: dict[str, np.ndarray]             # (T,3) positions
    object_yaw: dict[str, np.ndarray]          # (T,)
    events: list[dict]                         # {"type": grasp|release, "t": s, "obj": name}
    phases: list[tuple[str, float, float, str]]  # (phase, t0, t1, obj)
    cam: list[PhoneCamera]                     # per frame
    program_seen: list[tuple[str, np.ndarray]] = field(default_factory=list)  # (obj, place xyz)


@dataclass
class HumanVideo:
    rgb: np.ndarray       # (T,H,W,3) uint8
    depth: np.ndarray     # (T,H,W) float32 metres (LiDAR-like, noisy) -- zeros if RGB-only
    K: np.ndarray
    fps: int
    gt: DemoGT
    card: dict
    duration_s: float


@dataclass
class HumanStyle:
    speed: float
    approach_deg: float
    tremor: float
    pause: float
    side_bias: float

    @staticmethod
    def sample(rng: np.random.Generator) -> HumanStyle:
        return HumanStyle(speed=float(rng.uniform(0.4, 0.9)), approach_deg=float(rng.uniform(25, 60)),
                          tremor=float(rng.uniform(0.001, 0.002)), pause=float(rng.uniform(0.1, 0.35)),
                          side_bias=float(rng.uniform(-0.4, 0.4)))


def _seg(p0: np.ndarray, p1: np.ndarray, speed: float, arc: float = 0.0,
         min_t: float = 0.25) -> tuple[np.ndarray, float]:
    """Min-jerk segment sampled at FPS; returns (N,3) positions and duration."""
    d = float(np.linalg.norm(p1 - p0))
    T = max(min_t, 1.875 * d / max(speed, 1e-3))  # min-jerk peak speed = 1.875 * d / T
    n = max(2, int(round(T * FPS)))
    s = min_jerk(np.linspace(0, 1, n))[:, None]
    pts = p0[None] + s * (p1 - p0)[None]
    if arc:
        pts[:, 2] += arc * np.sin(np.pi * s[:, 0])
    return pts, T


class HumanDemonstrator:
    """Generates the hand trajectory + GT annotations for one chore demonstration."""

    def __init__(self, chore: ChoreSpec, env: TidyEnv, rng: np.random.Generator):
        self.chore, self.env, self.rng = chore, env, rng
        self.style = HumanStyle.sample(rng)

    def _hand_frame(self, center: np.ndarray, jaw_yaw: float, approach_deg: float, width: float) -> HandPose:
        """Thumb/index straddle `center` along jaw_yaw; wrist back toward the human (+x) and up."""
        u = np.array([np.cos(jaw_yaw), np.sin(jaw_yaw), 0.0])
        a = np.deg2rad(approach_deg)
        # approach direction: from the human side (+x), tilted from vertical by `a`
        horiz = np.array([1.0, self.style.side_bias, 0.0])
        horiz /= np.linalg.norm(horiz)
        back = np.sin(a) * horiz + np.cos(a) * np.array([0, 0, 1.0])
        thumb = center + 0.5 * width * u
        index = center - 0.5 * width * u
        wrist = center + 0.085 * back
        return HandPose(wrist, thumb, index)

    def _planned_goal(self, name: str, placed: dict[str, np.ndarray]) -> np.ndarray:
        """Goal of `name` given where this demonstration has already put other objects (the whole hand path is
        planned before rendering, so the env still shows every object at its start pose)."""
        g = self.chore.program[name]
        kind, lm = g.landmark.split(":", 1)
        if kind == "object" and lm in placed:
            return placed[lm] + np.array(g.offset)
        return goal_position(g, self.env)

    def generate(self) -> tuple[list[HandPose], list[dict], list[tuple[str, float, float, str]], list]:
        env, ch, st = self.env, self.chore, self.style
        poses: list[HandPose] = []
        events: list[dict] = []
        phases: list[tuple[str, float, float, str]] = []
        carried: list[tuple[int, str | None]] = []
        program_seen: list = []
        rest = HandPose(np.array([0.62, 0.0, 0.40]), np.array([0.56, 0.03, 0.36]), np.array([0.56, -0.03, 0.36]))
        cur = rest

        def emit(path: list[HandPose], phase: str, obj: str, holding: str | None) -> None:
            t0 = len(poses) / FPS
            for hp in path:
                trem = self.rng.normal(0, st.tremor, 3)
                poses.append(HandPose(hp.wrist + trem, hp.thumb + trem, hp.index + trem))
                carried.append((len(poses) - 1, holding))
            phases.append((phase, t0, len(poses) / FPS, obj))

        def move(a: HandPose, b: HandPose, speed: float, arc: float = 0.0) -> list[HandPose]:
            w, _ = _seg(a.wrist, b.wrist, speed, arc)
            th, _ = _seg(a.thumb, b.thumb, speed, arc)
            ix, _ = _seg(a.index, b.index, speed, arc)
            n = min(len(w), len(th), len(ix))
            return [HandPose(w[i], th[i], ix[i]) for i in range(n)]

        def hold(p: HandPose, secs: float) -> list[HandPose]:
            return [p] * max(1, int(round(secs * FPS)))

        emit(hold(rest, 0.8), "idle", "", None)
        placed: dict[str, np.ndarray] = {}
        remaining = [n for n in ch.objects if n in ch.program]
        order = list(ch.order or [])
        while remaining:
            if order:
                name = next(n for n in order if n in remaining)
            else:
                name = str(self.rng.choice(remaining))
            remaining.remove(name)
            o = env.obj_by_name[name]
            op, oy = env.obj_pose(name)
            jaw = grasp_yaw(o, oy) + self.rng.normal(0, np.deg2rad(4))
            w_obj = 2 * jaw_half_extent(o)
            gz = grasp_height(o) + self.rng.normal(0, 0.003)
            center = np.array([op[0], op[1], gz]) + np.r_[self.rng.normal(0, 0.003, 2), 0]
            pre = self._hand_frame(center + np.array([0, 0, 0.06]), jaw, st.approach_deg, w_obj + 0.04)
            at = self._hand_frame(center, jaw, st.approach_deg, w_obj + 0.03)
            pinch = self._hand_frame(center, jaw, st.approach_deg, w_obj)
            emit(move(cur, pre, st.speed, arc=0.04), "reach", name, None)
            emit(move(pre, at, st.speed * 0.5), "reach", name, None)
            emit(move(at, pinch, 0.15) + hold(pinch, st.pause), "grasp", name, None)
            events.append({"type": "grasp", "t": len(poses) / FPS, "obj": name})
            goal = self._planned_goal(name, placed)
            place_c = np.array([goal[0], goal[1], goal[2] + gz + 0.012]) + np.r_[self.rng.normal(0, 0.004, 2), 0]
            lift = self._hand_frame(center + np.array([0, 0, 0.08]), jaw, st.approach_deg, w_obj)
            above = self._hand_frame(place_c + np.array([0, 0, 0.06]), jaw, st.approach_deg * 0.7, w_obj)
            down = self._hand_frame(place_c, jaw, st.approach_deg * 0.7, w_obj)
            emit(move(pinch, lift, st.speed * 0.7), "transport", name, name)
            emit(move(lift, above, st.speed, arc=0.03), "transport", name, name)
            emit(move(above, down, st.speed * 0.4), "place", name, name)
            # release: open fingers
            opened = self._hand_frame(place_c, jaw, st.approach_deg * 0.7, w_obj + 0.03)
            events.append({"type": "release", "t": len(poses) / FPS, "obj": name})
            emit(move(down, opened, 0.1) + hold(opened, st.pause * 0.5), "release", name, None)
            up = self._hand_frame(place_c + np.array([0, 0, 0.08]), jaw, st.approach_deg, w_obj + 0.03)
            emit(move(opened, up, st.speed * 0.7), "retreat", name, None)
            program_seen.append((name, place_c.copy()))
            placed[name] = np.array([place_c[0], place_c[1], goal[2]])  # base of the object after placing
            cur = up
            # simulate the object being carried (for rendering + GT): done in render loop via `carried`
            self._pending = None
        emit(move(cur, rest, st.speed), "idle", "", None)
        emit(hold(rest, 0.8), "idle", "", None)
        self.carried = carried
        return poses, events, phases, program_seen


def sample_phone(rng: np.random.Generator) -> PhoneCamera:
    pos = np.array([0.18 + rng.uniform(0.38, 0.50), rng.uniform(-0.12, 0.12), rng.uniform(0.38, 0.50)])
    look = np.array([0.16 + rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0.0])
    return PhoneCamera(pos, look, float(rng.normal(0, np.deg2rad(2))))


def _set_mocap(env: TidyEnv, hp: HandPose) -> None:
    m, d = env.model, env.data
    for name, p in (("h_wrist", hp.wrist), ("h_thumb", hp.thumb), ("h_index", hp.index)):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        d.mocap_pos[m.body_mocapid[bid]] = p
    # finger segments: capsules from wrist-ish knuckle toward the tips (visual only)
    knuckle = hp.wrist + 0.45 * ((hp.thumb + hp.index) / 2 - hp.wrist)
    for seg, tip in (("h_thumb_seg", hp.thumb), ("h_index_seg", hp.index)):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, seg)
        mid = 0.5 * (knuckle + tip)
        v = tip - knuckle
        v = v / (np.linalg.norm(v) + 1e-9)
        z = np.array([0, 0, 1.0])
        ax = np.cross(z, v)
        s = np.linalg.norm(ax)
        half = np.arcsin(min(1, s)) / 2
        q = np.array([1.0, 0, 0, 0]) if s < 1e-6 else np.r_[np.cos(half), ax / s * np.sin(half)]
        if np.dot(z, v) < 0 and s >= 1e-6:
            ang = np.pi - np.arcsin(min(1, s))
            q = np.r_[np.cos(ang / 2), ax / s * np.sin(ang / 2)]
        d.mocap_pos[m.body_mocapid[bid]] = mid
        d.mocap_quat[m.body_mocapid[bid]] = q
    # wrist orientation: forearm capsule points from wrist toward the human (+x, up)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "h_wrist")
    fwd = hp.wrist - knuckle
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    d.mocap_quat[m.body_mocapid[bid]] = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def record_demo(chore: ChoreSpec, card: PlacementCard, seed: int, depth_noise: float = 0.004,
                with_depth: bool = True, shake_deg: float = 0.4) -> HumanVideo:
    """Simulate a person doing the chore while filming with a phone; returns frames + GT."""
    rng = np.random.default_rng(seed)
    light, tint = LIGHTING[card.lighting]
    env = TidyEnv(chore.object_specs(card.subs), chore.fixture_specs(), include_hand=True,
                  light_scale=light, light_tint=tint)
    env.reset(card.placement)
    demo = HumanDemonstrator(chore, env, rng)
    poses, events, phases, program_seen = demo.generate()
    cam0 = sample_phone(rng)
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "phone")
    renderer = mujoco.Renderer(env.model, PHONE_H, PHONE_W)
    rgbs, depths, cams = [], [], []
    T = len(poses)
    obj_tr = {n: np.zeros((T, 3)) for n in chore.objects}
    obj_yaw = {n: np.zeros(T) for n in chore.objects}
    holding_prev: str | None = None
    rel: tuple[np.ndarray, float] | None = None
    shake = np.zeros(3)
    for i, hp in enumerate(poses):
        holding = demo.carried[i][1]
        _set_mocap(env, hp)
        center = 0.5 * (hp.thumb + hp.index)
        if holding is not None:
            if holding_prev != holding:
                op, oy = env.obj_pose(holding)
                rel = (op - center, oy)
            assert rel is not None
            env.set_obj_pose(holding, (center[0] + rel[0][0], center[1] + rel[0][1]), rel[1],
                             z=float(center[2] + rel[0][2]))
        elif holding_prev is not None:
            pass  # released: physics takes over
        holding_prev = holding
        for _ in range(int(round(1 / FPS / 0.002))):
            mujoco.mj_step(env.model, env.data)
            if holding is not None:
                env.set_obj_pose(holding, (center[0] + rel[0][0], center[1] + rel[0][1]), rel[1],  # type: ignore[index]
                                 z=float(center[2] + rel[0][2]))  # type: ignore[index]
        mujoco.mj_forward(env.model, env.data)
        shake = 0.85 * shake + rng.normal(0, np.deg2rad(shake_deg) * 0.3, 3)
        cam = PhoneCamera(cam0.pos + 0.004 * shake, cam0.look_at + 0.01 * shake, cam0.up_roll + shake[2])
        env.model.cam_pos[cam_id] = cam.pos
        ax = cam.xyaxes()
        R = np.stack([ax[:3], ax[3:], np.cross(ax[:3], ax[3:])], axis=1)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        env.model.cam_quat[cam_id] = q
        env.model.cam_fovy[cam_id] = PHONE_FOVY
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, "phone")
        rgbs.append(renderer.render().copy())
        if with_depth:
            renderer.enable_depth_rendering()
            renderer.update_scene(env.data, "phone")
            dimg = renderer.render().copy()
            renderer.disable_depth_rendering()
            dimg = dimg + rng.normal(0, depth_noise, dimg.shape).astype(np.float32) * (dimg / 0.6)
            depths.append(dimg.astype(np.float32))
        cams.append(cam)
        for n in chore.objects:
            p, y = env.obj_pose(n)
            obj_tr[n][i] = p
            obj_yaw[n][i] = y
    renderer.close()
    env.close()
    t = np.arange(T) / FPS
    gt = DemoGT(t, {"wrist": np.array([p.wrist for p in poses]), "thumb": np.array([p.thumb for p in poses]),
                    "index": np.array([p.index for p in poses])}, obj_tr, obj_yaw, events, phases, cams, program_seen)
    depth = np.stack(depths) if with_depth else np.zeros((T, PHONE_H, PHONE_W), np.float32)
    return HumanVideo(np.stack(rgbs), depth, cam0.intrinsics(), FPS, gt, card.to_json(), T / FPS)


def jaw_yaw_of(hp: HandPose) -> float:
    v = hp.thumb - hp.index
    return float(wrap_half_pi(np.arctan2(v[1], v[0])))
