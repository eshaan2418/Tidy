"""Hand -> parallel-gripper retargeting and pseudo-demo construction (Contribution 1, part B).

For each observed pick-place segment:
  1. grasp centre  = midpoint of thumb & index tips; jaw yaw = thumb-index axis (pi-symmetric);
     opening = thumb-index distance minus the fingertip thickness (calibrated constant).
  2. robot-feasible approach: the human approaches obliquely from their side; the robot instead
     descends vertically over the last 6 cm (approach correction) and retreats vertically after release.
  3. clearance lifting: any long horizontal move happens at >= TRANSIT_Z (fingers clear objects).
  4. IK feasibility filter (closed-form IK with pitch search); segments with >20 % unreachable
     waypoints are rejected.
  5. temporal smoothing + arc-length time scaling to the robot's speed limit (10 Hz ticks), with dwell
     ticks for the rate-limited gripper.
`hand` mode (ablation "-object-centric"): grasp/place targets from the hand keypoints only (no snapping to the
tracked object pose, D-020).
`raw` mode (ablation "-retargeting") uses the wrist trajectory as the TCP, keeps the human timing
profile (scaled), and derives jaw yaw from the hand heading -- i.e., no embodiment-aware mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

from common.geometry import wrap_half_pi
from common.kinematics import GEOM, ik, preferred_pitch
from phases.events import PickPlace, Segmentation
from policies.spec import View, action_vector, obs_vector
from safety.limits import DEFAULT_SAFETY
from sim.env import EEAction
from sim.expert import TRANSIT_Z, jaw_half_extent, polar_step
from sim.scene import ObjectSpec

FINGERTIP_THICKNESS = 0.017   # thumb-index centre distance at contact minus object width (cap radii)
ROBOT_STEP = 0.016            # m per 10 Hz tick (< 0.018 used by the safety-limited controller)


@dataclass
class Waypoint:
    pos: np.ndarray
    yaw: float
    pitch: float
    grip: float


@dataclass
class RetargetedSegment:
    obj: str
    waypoints: list[Waypoint]
    pick_pos: np.ndarray
    pick_yaw: float
    place_pos: np.ndarray
    feasible_frac: float
    accepted: bool
    reason: str = ""
    human_duration_s: float = 0.0
    meta: dict = field(default_factory=dict)


def _smooth(x: np.ndarray, n: int = 5) -> np.ndarray:
    ok = ~np.isnan(x[:, 0])
    y = x.copy()
    if ok.sum() > n:
        y[ok] = uniform_filter1d(x[ok], n, axis=0, mode="nearest")
    return y


def _smooth_yaw(yaw: np.ndarray, n: int = 7) -> np.ndarray:
    ok = ~np.isnan(yaw)
    out = yaw.copy()
    if ok.sum() > n:
        c, s = uniform_filter1d(np.cos(2 * yaw[ok]), n), uniform_filter1d(np.sin(2 * yaw[ok]), n)
        out[ok] = 0.5 * np.arctan2(s, c)
    return out


def opening_to_cmd(opening: float) -> float:
    return float(np.clip(1.0 - opening / GEOM.max_opening, 0.0, 1.0))


def _path_to(cur: Waypoint, goal: np.ndarray, yaw: float, grip: float, transit: bool) -> list[Waypoint]:
    out = []
    p = cur.pos.copy()
    for _ in range(200):
        if np.linalg.norm(goal - p) < 1e-3:
            break
        p = polar_step(p, goal, ROBOT_STEP, TRANSIT_Z if transit else -1.0)
        out.append(Waypoint(p.copy(), yaw, preferred_pitch(p), grip))
    return out


def _resample_human(c: np.ndarray, yaw: np.ndarray, grip: np.ndarray) -> list[Waypoint]:
    """Arc-length resampling of a human path at robot speed (keeps the human's spatial path)."""
    ok = ~np.isnan(c[:, 0])
    c, yaw, grip = c[ok], yaw[ok], grip[ok]
    if len(c) < 2:
        return []
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(2, int(np.ceil(s[-1] / ROBOT_STEP)) + 1)
    q = np.linspace(0, s[-1], n)
    out = []
    for qq in q:
        j = int(np.clip(np.searchsorted(s, qq) - 1, 0, len(c) - 2))
        a = 0.0 if s[j + 1] == s[j] else (qq - s[j]) / (s[j + 1] - s[j])
        p = (1 - a) * c[j] + a * c[j + 1]
        out.append(Waypoint(p, float(yaw[j]), preferred_pitch(p), float(grip[j])))
    return out


def object_centric_grasp(gpos: np.ndarray, gyaw: float, pk: PickPlace, obj: ObjectSpec) -> tuple[np.ndarray, float]:
    """Fuse the (noisy) hand grasp pose with the tracked object pose (D-020): grasp xy = object rest xy when the
    hand is within 3 cm of it; height clipped to the object's body; for elongated objects the jaw axis snaps to
    the object axis (major or minor) closest to the hand's thumb-index axis, so the human still chooses it."""
    g = gpos.copy()
    if not np.any(np.isnan(pk.pick_pos)) and np.linalg.norm(pk.pick_pos[:2] - g[:2]) < 0.03:
        g[:2] = pk.pick_pos[:2]
    g[2] = float(np.clip(g[2], 0.35 * obj.height, max(0.65 * obj.height, 0.014)))
    yaw = gyaw
    if obj.kind in ("box", "capsule") and not np.isnan(pk.pick_yaw):
        cands = [pk.pick_yaw, pk.pick_yaw + np.pi / 2]
        yaw = float(min(cands, key=lambda a: abs(float(wrap_half_pi(a - gyaw)))))
        yaw = float(wrap_half_pi(yaw))
    return g, yaw


def retarget_pick(seg: Segmentation, pk: PickPlace, obj: ObjectSpec, fps: int, mode: str = "full",
                  start: Waypoint | None = None) -> RetargetedSegment:
    T = len(seg.thumb)
    i0 = int(round(pk.t_segment_start * fps))
    ig = int(round(pk.t_grasp * fps))
    ir = int(round(pk.t_release * fps))
    i1 = min(T, ir + int(0.5 * fps))
    th, ix, wr = seg.thumb, seg.index, seg.wrist
    c = _smooth(0.5 * (th + ix))
    v = th - ix
    yaw = _smooth_yaw(np.asarray(wrap_half_pi(np.arctan2(v[:, 1], v[:, 0]))))
    ap = np.linalg.norm(v, axis=1)
    open_w = np.clip(ap - FINGERTIP_THICKNESS, 0.0, GEOM.max_opening)
    half = jaw_half_extent(obj)
    floor = DEFAULT_SAFETY.workspace.lo[2]
    wps: list[Waypoint] = []
    cur = start or Waypoint(np.array([0.10, 0.0, 0.14]), 0.0, 0.0, 0.0)
    if mode == "raw":
        # raw hand trajectory: wrist as TCP, heading-derived yaw, human-timed gripper (threshold on aperture)
        w = _smooth(np.where(np.isnan(wr), c, wr))
        head = c - w
        hyaw = np.asarray(wrap_half_pi(np.arctan2(head[:, 1], head[:, 0])))
        grip = np.where((np.arange(T) >= ig) & (np.arange(T) < ir), 1.0, 0.0)
        w = w.copy()
        w[:, 2] = np.maximum(w[:, 2], floor)
        wps = _resample_human(w[i0:i1], hyaw[i0:i1], grip[i0:i1])
    else:
        # grasp pose from the hand at the grasp event
        g_win = slice(max(0, ig - 3), min(T, ig + 2))
        gpos = np.nanmedian(c[g_win], axis=0)
        gyaw = float(np.nanmedian(yaw[g_win])) if np.any(~np.isnan(yaw[g_win])) else pk.pick_yaw + np.pi / 2
        if np.any(np.isnan(gpos)):
            gpos = pk.pick_pos.copy()
        if mode == "full":
            gpos, gyaw = object_centric_grasp(gpos, gyaw, pk, obj)
        gpos[2] = max(gpos[2], floor + 0.001)
        pre_open = float(np.clip(np.nanmax(open_w[max(i0, ig - fps):ig + 1]) if ig > i0 else 2 * half + 0.03,
                                 2 * half + 0.025, GEOM.max_opening))
        g_open = opening_to_cmd(pre_open)
        # 1) reach: human path at robot speed until ~6 cm above the grasp point, with clearance lifting
        above = gpos + np.array([0, 0, 0.06])
        wps += _path_to(cur, above, gyaw, g_open, transit=True)
        last = wps[-1] if wps else cur
        # 2) vertical descent (approach correction)
        wps += _path_to(last, gpos, gyaw, g_open, transit=False)
        last = wps[-1] if wps else cur
        # 3) grasp dwell (gripper rate-limited)
        wps += [Waypoint(gpos.copy(), gyaw, preferred_pitch(gpos), 1.0) for _ in range(8)]
        # 4) transport: follow the human's carry path in xy, lifted to clearance height, at robot speed
        rpos = np.nanmedian(c[max(ig, ir - 3):ir + 2], axis=0) if ir > ig else pk.place_pos + np.array([0, 0, 0.03])
        if np.any(np.isnan(rpos)):
            rpos = pk.place_pos + np.array([0, 0, obj.height / 2 + 0.01])
        if mode == "full" and not np.any(np.isnan(pk.place_pos)):
            # object-centric place target: where the object came to rest (tracked) in xy, hand release height
            if np.linalg.norm(pk.place_pos[:2] - rpos[:2]) < 0.05:
                rpos[:2] = pk.place_pos[:2]
        rpos[2] = max(rpos[2], floor + 0.001)
        lift = gpos + np.array([0, 0, 0.0])
        lift[2] = max(TRANSIT_Z, gpos[2] + 0.05)
        wps += _path_to(Waypoint(gpos, gyaw, 0.0, 1.0), lift, gyaw, 1.0, transit=False)
        carry = c[ig:ir].copy()
        carry[:, 2] = np.maximum(carry[:, 2], TRANSIT_Z)
        mid = [p for p in carry[int(0.3 * len(carry)):int(0.7 * len(carry))] if not np.any(np.isnan(p))]
        via = list(mid[:: max(1, len(mid) // 3)]) if mid else []
        p_last = wps[-1]
        for vpt in via + [rpos + np.array([0, 0, max(0.0, TRANSIT_Z - rpos[2])])]:
            wps += _path_to(p_last, np.asarray(vpt), gyaw, 1.0, transit=False)
            p_last = wps[-1] if wps else p_last
        # 5) lower to the release point
        wps += _path_to(p_last, rpos, gyaw, 1.0, transit=False)
        # 6) release dwell with a small opening (as the human did), 7) vertical retreat
        rel_open = float(np.clip(np.nanmedian(open_w[ir:min(T, ir + int(0.4 * fps))]) if ir < T else 2 * half + 0.02,
                                 2 * half + 0.008, GEOM.max_opening))
        r_cmd = opening_to_cmd(rel_open)
        wps += [Waypoint(rpos.copy(), gyaw, preferred_pitch(rpos), r_cmd) for _ in range(6)]
        up = rpos + np.array([0, 0, 0.0])
        up[2] = TRANSIT_Z
        wps += _path_to(wps[-1], up, gyaw, r_cmd, transit=False)
    # feasibility filter
    feas = []
    for wp in wps:
        r = ik(wp.pos, wp.yaw, wp.pitch)
        feas.append(r.pos_error < 1e-6)
        wp.pitch = r.pitch_used
    ff = float(np.mean(feas)) if feas else 0.0
    accepted = ff >= 0.8 and len(wps) > 5
    return RetargetedSegment(pk.obj, wps, pk.pick_pos, pk.pick_yaw, pk.place_pos, ff, accepted,
                             "" if accepted else f"feasible fraction {ff:.2f}", (i1 - i0) / fps,
                             {"mode": mode})


# ------------------------------------------------------------------------------------------ pseudo-demos
def pseudo_demo_pairs(rs: RetargetedSegment, obj: ObjectSpec, goal_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(obs, action) pairs in the shared spec, assuming the robot tracks the waypoints. The gripper opening
    proprio is simulated with the safety-filter rate limit and stops at the object width when closing."""
    obs, act = [], []
    opening = GEOM.max_opening * (1 - rs.waypoints[0].grip)
    grip_cmd = rs.waypoints[0].grip
    width = 2 * jaw_half_extent(obj)
    holding = False
    for k in range(len(rs.waypoints) - 1):
        w, nxt = rs.waypoints[k], rs.waypoints[k + 1]
        v = View(w.pos, w.yaw, w.pitch, opening, grip_cmd, rs.pick_pos, rs.pick_yaw, obj, goal_pos)
        obs.append(obs_vector(v))
        act.append(action_vector(w.pos, EEAction(nxt.pos, nxt.yaw, nxt.pitch, nxt.grip)))
        # proprio update (rate-limited like the safety filter)
        g_step = DEFAULT_SAFETY.max_gripper_vel * 0.1 / GEOM.max_opening
        grip_cmd = float(np.clip(nxt.grip, grip_cmd - g_step, grip_cmd + g_step))
        target_open = (1 - grip_cmd) * GEOM.max_opening
        if nxt.grip > 0.9 and abs(opening - width) < 0.02:
            holding = True
        if holding and target_open > width + 0.004:
            holding = False
        opening = max(target_open, width) if holding or (target_open < width and nxt.grip > 0.9) else target_open
    return np.array(obs, np.float32), np.array(act, np.float32)
