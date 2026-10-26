"""MuJoCo tabletop environment for the SO-100-class arm (approximate digital twin).

Control: 10 Hz end-effector commands -> analytic IK -> SafeController (same safety code as the
real robot) -> joint position servos simulated at 500 Hz.

Grasping simplification (documented in DECISIONS.md): fingers collide physically with objects,
but a grasp "holds" via kinematic attachment when the jaws close on an object that is
geometrically inside the jaws (lateral offset, jaw-axis extent vs opening, and height checks).
This is a standard sim shortcut; it makes grasp *precision* matter but not grasp *force*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from common.geometry import quat_yaw, wrap_half_pi, yaw_quat
from common.kinematics import GEOM, HOME_Q, fk, ik
from safety.core import GuardedDriver, LatchingEStop, ManualEStop, SafeController, Watchdog
from safety.limits import DEFAULT_SAFETY, SafetyConfig
from sim.scene import FixtureSpec, ObjectSpec, SceneSpec, build_xml

CONTROL_DT = 0.1
PHYS_DT = 0.002
N_SUB = int(round(CONTROL_DT / PHYS_DT))
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@dataclass
class EEAction:
    """Policy action: TCP position target, jaw yaw (pi-symmetric), tool pitch, gripper (0=open,1=closed)."""
    pos: np.ndarray
    yaw: float
    pitch: float
    gripper: float


@dataclass
class Placement:
    """Object poses on the table: name -> (x, y, yaw)."""
    poses: dict[str, tuple[float, float, float]]
    meta: dict = field(default_factory=dict)


class SimDriver(GuardedDriver):
    def __init__(self, env: TidyEnv):
        super().__init__()
        self.env = env
        self.frozen = False

    def read_state(self) -> tuple[np.ndarray, float]:
        return self.env.q_cmd.copy(), self.env.g_cmd

    def send(self, token: object, q: np.ndarray, gripper: float) -> None:
        self._check(token)
        self.frozen = False
        self.env.q_cmd = np.asarray(q, dtype=float).copy()
        self.env.g_cmd = float(gripper)

    def freeze(self, token: object) -> None:
        self._check(token)
        self.frozen = True
        self.env.q_cmd = self.env.joint_pos()
        # gripper keeps its last command (do not drop held objects on a freeze)


class TidyEnv:
    def __init__(self, objects: list[ObjectSpec], fixtures: list[FixtureSpec], include_hand: bool = False,
                 light_scale: float = 1.0, light_tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 safety_cfg: SafetyConfig = DEFAULT_SAFETY, distractors: list[ObjectSpec] | None = None):
        self.objects = list(objects)
        self.distractors = list(distractors or [])
        self.fixtures = list(fixtures)
        self.spec = SceneSpec(self.objects, self.fixtures, include_hand, light_scale, light_tint, self.distractors)
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.spec))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._jq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
        self._fq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ("finger_l", "finger_r")]
        self._tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.all_objects = self.objects + self.distractors
        self._oq: dict[str, int] = {}
        self._ov: dict[str, int] = {}
        for o in self.all_objects:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{o.name}_jnt")
            self._oq[o.name] = m.jnt_qposadr[jid]
            self._ov[o.name] = m.jnt_dofadr[jid]
        self.obj_by_name = {o.name: o for o in self.all_objects}
        self._weld = {o.name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip_{o.name}")
                      for o in self.all_objects}
        self._obody = {o.name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{o.name}")
                       for o in self.all_objects}
        self.fix_by_name = {f.name: f for f in self.fixtures}
        self.include_hand = include_hand
        # safety
        self.estop_button = ManualEStop()
        self._sim_time = 0.0
        self.watchdog = Watchdog(DEFAULT_SAFETY.watchdog_timeout_s, clock=lambda: self._sim_time)
        self.driver = SimDriver(self)
        self.safety = SafeController(self.driver, LatchingEStop([self.estop_button]), self.watchdog, None, safety_cfg)
        self.q_cmd = HOME_Q.copy()
        self.g_cmd = 0.0
        self.attached: str | None = None
        self._held_width = 0.0
        self._rel: tuple[np.ndarray, np.ndarray] | None = None
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.step_count = 0
        self.safety_events: list[str] = []

    # ----------------------------------------------------------------------------------- state access
    def joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[i] for i in self._jq])

    def opening(self) -> float:
        return float(self.data.qpos[self._fq[0]] + self.data.qpos[self._fq[1]])

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[self._tcp].copy(), self.data.site_xmat[self._tcp].reshape(3, 3).copy()

    def ee_state(self) -> tuple[np.ndarray, float, float]:
        return fk(self.joint_pos())

    def obj_pose(self, name: str) -> tuple[np.ndarray, float]:
        a = self._oq[name]
        return self.data.qpos[a:a + 3].copy(), quat_yaw(self.data.qpos[a + 3:a + 7])

    def obj_upright(self, name: str) -> bool:
        a = self._oq[name]
        q = self.data.qpos[a + 3:a + 7]
        # z-axis of object frame vs world z
        zz = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        return bool(zz > 0.9)

    def set_obj_pose(self, name: str, xy: tuple[float, float], yaw: float, z: float | None = None) -> None:
        o = self.obj_by_name[name]
        a = self._oq[name]
        zz = (o.height / 2 + 0.0005) if z is None else z
        self.data.qpos[a:a + 3] = [xy[0], xy[1], zz]
        self.data.qpos[a + 3:a + 7] = yaw_quat(yaw)
        v = self._ov[name]
        self.data.qvel[v:v + 6] = 0

    # ----------------------------------------------------------------------------------- reset / step
    def reset(self, placement: Placement, settle: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for i, q in zip(self._jq, HOME_Q):
            self.data.qpos[i] = q
        self.q_cmd = HOME_Q.copy()
        self.g_cmd = 0.0
        self.attached, self._rel = None, None
        self.data.eq_active[:] = 0
        park = 0
        for o in self.all_objects:
            if o.name in placement.poses:
                x, y, yaw = placement.poses[o.name]
                self.set_obj_pose(o.name, (x, y), yaw)
            else:  # park unused objects far away
                self.set_obj_pose(o.name, (-0.5 - 0.1 * park, 0.6), 0.0)
                park += 1
        self._apply_ctrl()
        mujoco.mj_forward(self.model, self.data)
        self.safety.estop.tripped = False
        self.estop_button.pressed = False
        self.watchdog.heartbeat()
        self.step_count = 0
        self.safety_events = []
        if settle:
            for _ in range(50):
                self._substep()

    def _apply_ctrl(self) -> None:
        self.data.ctrl[:5] = self.q_cmd
        half = (1.0 - self.g_cmd) * GEOM.max_opening / 2
        self.data.ctrl[5] = half
        self.data.ctrl[6] = half

    def _substep(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self._sim_time += PHYS_DT
        # stabilisation (D-008): clamp horizontal / angular speed of free objects. Squeezing a light object
        # between a stiff servo and the table otherwise ejects it at unphysical speed (MuJoCo soft contacts).
        for v in self._ov.values():
            h = self.data.qvel[v:v + 2]
            n = float(np.hypot(h[0], h[1]))
            if n > 0.35:
                self.data.qvel[v:v + 2] = h / n * 0.35
            ang = self.data.qvel[v + 3:v + 6]
            na = float(np.linalg.norm(ang))
            if na > 10.0:
                self.data.qvel[v + 3:v + 6] = ang / na * 10.0

    def _attach(self, name: str) -> None:
        """Activate the hand->object weld at the current relative pose (D-003, revised by D-008)."""
        hb, ob = self._hand, self._obody[name]
        R1 = self.data.xmat[hb].reshape(3, 3)
        R2 = self.data.xmat[ob].reshape(3, 3)
        rp = R1.T @ (self.data.xpos[ob] - self.data.xpos[hb])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R1.T @ R2).flatten())
        self._held_width = self._jaw_width(name)
        e = self._weld[name]
        self.model.eq_data[e, 0:3] = 0.0
        self.model.eq_data[e, 3:6] = rp
        self.model.eq_data[e, 6:10] = q
        self.model.eq_data[e, 10] = 1.0
        self.data.eq_active[e] = 1
        self.attached = name

    def _jaw_width(self, name: str) -> float:
        o = self.obj_by_name[name]
        _, R = self.tcp_pose()
        jaw_yaw = float(np.arctan2(R[1, 1], R[0, 1]))
        if o.kind == "cylinder" and not self.obj_upright(name):
            return 2 * o.size[1]
        return 2 * o.half_extent_along(self.obj_pose(name)[1], jaw_yaw)

    def _detach(self) -> None:
        if self.attached is not None:
            self.data.eq_active[self._weld[self.attached]] = 0
        self.attached, self._rel = None, None

    def grasp_check(self, name: str) -> bool:
        """Geometric jaw-containment test for attaching `name` (see module docstring)."""
        o = self.obj_by_name[name]
        p, R = self.tcp_pose()
        op, oyaw = self.obj_pose(name)
        local = R.T @ (op - p)          # tcp frame: x lateral(finger width dir), y jaw axis, z approach
        jaw_world = R[:, 1]
        jaw_yaw = float(np.arctan2(jaw_world[1], jaw_world[0]))
        if o.kind == "cylinder" and not self.obj_upright(name):
            ext = o.size[1]
        else:
            ext = o.half_extent_along(oyaw, jaw_yaw)
        open_now = self.opening()
        if 2 * ext > GEOM.max_opening - 0.002:
            return False
        if abs(local[1]) > max(0.0, open_now / 2 - ext) + 0.012:
            return False
        if abs(local[0]) > 0.010 + 0.5 * o.half_extent_along(oyaw, jaw_yaw + np.pi / 2):
            return False
        # height: object centre must be within the finger pad span along approach axis
        if local[2] < -0.034 or local[2] > 0.014:
            return False
        return True

    def step(self, act: EEAction, monitor_rgb: np.ndarray | None = None) -> dict:
        """One 10 Hz control step. Returns info dict (safety status, grasp events)."""
        q_now = self.q_cmd.copy()
        res = ik(np.asarray(act.pos, dtype=float), float(act.yaw), float(act.pitch), q_ref=q_now)
        # choose roll branch closest to current (pi symmetry of a parallel jaw)
        q_t = res.q.copy()
        alt = q_t[4] + (np.pi if q_t[4] < 0 else -np.pi)
        if abs(alt - q_now[4]) < abs(q_t[4] - q_now[4]) and abs(alt) <= GEOM.q_max[4]:
            q_t[4] = alt
        self.watchdog.heartbeat()
        status = self.safety.tick(q_t, float(np.clip(act.gripper, 0, 1)), CONTROL_DT, monitor_rgb)
        for e in status.events:
            if e.kind in ("workspace", "nan"):
                self.safety_events.append(f"{e.kind}:{e.detail}")
        if not status.sent:
            self.safety_events.append(status.reason)
        info = {"status": status.reason, "ik_feasible": res.feasible, "ik_pos_err": res.pos_error,
                "grasped": None, "released": None}
        target_open = (1.0 - self.g_cmd) * GEOM.max_opening
        for k in range(N_SUB):
            self._apply_ctrl()
            self._substep()
            if self.attached is None and k % 5 == 4 and target_open < self.opening() - 0.003:
                for o in self.objects:
                    # the command must squeeze the object (target narrower than the object) to hold it
                    if target_open < self._jaw_width(o.name) - 0.004 and self.grasp_check(o.name):
                        self._attach(o.name)
                        info["grasped"] = o.name
                        break
            if self.attached is not None and target_open > self._held_width + 0.004:
                info["released"] = self.attached
                self._detach()
        self.step_count += 1
        return info

    def settle(self, steps: int = 100) -> None:
        for _ in range(steps):
            self._apply_ctrl()
            self._substep()

    # ----------------------------------------------------------------------------------- rendering
    def render(self, camera: str = "overhead", width: int = 160, height: int = 120, depth: bool = False):
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height, width)
        r = self._renderers[key]
        r.update_scene(self.data, camera)
        rgb = r.render()
        if not depth:
            return rgb
        r.enable_depth_rendering()
        r.update_scene(self.data, camera)
        d = r.render().copy()
        r.disable_depth_rendering()
        return rgb, d

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers = {}


def yaw_of_jaw(env: TidyEnv) -> float:
    _, R = env.tcp_pose()
    return float(wrap_half_pi(np.arctan2(R[1, 1], R[0, 1])))
