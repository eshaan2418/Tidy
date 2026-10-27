"""Keyframe-based corrective augmentation (D-021), applied identically to EVERY data source (teleop, video,
practice, sim pre-training), so it never favours one method.

Problem it addresses: with 10-40 demonstrated object steps, the policy only sees states on the demonstrated
paths.  A small error early in the reach (e.g., descending 4 cm short of the object) leads to states it has
never seen, and it closes the jaw on air (traced in M4 debugging, see DECISIONS.md D-021).

Method (DART/MimicGen-flavoured, but from the demos alone; no expert queries):
  1. keyframes are read off each segment's own action stream: `close` = first tick the gripper command exceeds
     0.8, `open` = first tick after that it drops below 0.8.  `descend start` / `lower start` = the last tick
     before close/open at which the TCP is still >= 3 cm above (or >= 1 cm beside) the keyframe.
  2. a synthetic start state is drawn around the demonstrated reach (or carry) path (Gaussian, sigma 3 cm).
  3. a bounded-speed path in (radius, pan, z) goes from that state to the demo's descend/lower start point,
     and the *demonstrated* actions are used from there on (so the grasp/release itself is always the human's).
Only the synthetic part of each new segment produces training samples (`meta["n_valid"]`); its action chunks
run on into the demonstrated suffix.
"""

from __future__ import annotations

import numpy as np

from common.kinematics import preferred_pitch
from policies.data import EE_POS, GOAL_REL, OBJ_REL, Segment
from policies.spec import DELTA_SCALE
from sim.expert import TRANSIT_Z, polar_step

STEP = 0.016


def keyframes(seg: Segment) -> dict[str, int] | None:
    g = seg.act[:, 6]
    closes = np.nonzero(g > 0.8)[0]
    if len(closes) == 0:
        return None
    c = int(closes[0])
    opens = np.nonzero(g[c:] < 0.8)[0]
    o = int(c + opens[0]) if len(opens) else len(g) - 1
    ee = seg.obs[:, EE_POS] / 5.0

    def start_of(k: int, lo: int) -> int:
        for t in range(k, lo - 1, -1):
            if ee[t, 2] - ee[k, 2] >= 0.03 or np.linalg.norm(ee[t, :2] - ee[k, :2]) >= 0.01:
                return t
        return lo

    return {"close": c, "open": o, "descend": start_of(c, 0), "lower": start_of(o, c)}


def _obs_at(template: np.ndarray, ee_old: np.ndarray, ee_new: np.ndarray) -> np.ndarray:
    """Re-express an observation for a different TCP position (object/goal are fixed in the table frame)."""
    o = template.copy()
    obj = o[OBJ_REL] / 10.0 + ee_old
    goal = o[GOAL_REL] / 10.0 + ee_old
    o[EE_POS] = ee_new * 5.0
    o[OBJ_REL] = (obj - ee_new) * 10.0
    o[GOAL_REL] = (goal - ee_new) * 10.0
    o[5] = preferred_pitch(ee_new)
    return o


def _synth(seg: Segment, t_src: int, t_join: int, start: np.ndarray) -> Segment | None:
    ee = seg.obs[:, EE_POS] / 5.0
    target = ee[t_join]
    tmpl, a_tmpl = seg.obs[t_src], seg.act[t_join]
    path = [start]
    p = start.copy()
    for _ in range(80):
        if np.linalg.norm(target - p) < 1e-3:
            break
        p = polar_step(p, target, STEP, TRANSIT_Z if t_src < t_join else -1.0)
        path.append(p.copy())
    if len(path) < 2 or np.linalg.norm(path[-1] - target) > 2e-3:
        return None
    obs_l, act_l = [], []
    for k in range(len(path) - 1):
        o = _obs_at(tmpl, ee[t_src], path[k])
        a = a_tmpl.copy()
        a[:3] = np.clip((path[k + 1] - path[k]) / DELTA_SCALE, -1.5, 1.5)
        a[5] = preferred_pitch(path[k + 1])
        obs_l.append(o)
        act_l.append(a)
    n_valid = len(obs_l)
    obs = np.concatenate([np.array(obs_l, np.float32), seg.obs[t_join:]], 0)
    act = np.concatenate([np.array(act_l, np.float32), seg.act[t_join:]], 0)
    return Segment(obs, act, seg.source, seg.obj, {**seg.meta, "n_valid": n_valid, "synthetic": True})


def augment(segs: list[Segment], n_per: int = 6, sigma: float = 0.03, seed: int = 0) -> list[Segment]:
    """Return segs + synthetic corrective segments (n_per for reach, n_per for carry, per segment)."""
    rng = np.random.default_rng(seed)
    out = list(segs)
    for s in segs:
        kf = keyframes(s)
        if kf is None or len(s.obs) < 6:
            continue
        ee = s.obs[:, EE_POS] / 5.0
        for _phase, lo, join in (("reach", 0, kf["descend"]), ("carry", kf["close"] + 1, kf["lower"])):
            if join <= lo:
                continue
            for _ in range(n_per):
                t = int(rng.integers(lo, join + 1))
                start = ee[t] + rng.normal(0, sigma, 3) * np.array([1, 1, 0.6])
                start[2] = float(np.clip(start[2], 0.06, 0.18))
                start[0] = float(np.clip(start[0], 0.0, 0.30))
                start[1] = float(np.clip(start[1], -0.28, 0.28))
                syn = _synth(s, t, join, start)
                if syn is not None:
                    out.append(syn)
    return out
