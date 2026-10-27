"""Real phone clips -> processed demo (no simulator ground truth). Same perception, phase and retargeting code as
the sim pipeline (perception/video_dataset.py); hand keypoints from the fingertip-cap detector (`caps`) until a
learned 3D hand-pose model is plugged in (BLOCKERS B-005).

Clip directory format (convert phone exports to this; e.g. Record3D -> rgb.mp4 + depth.npz):
  rgb.mp4            RGB video
  depth.npz          optional, key "depth": (T, H, W) metres aligned with rgb (LiDAR phones)
  intrinsics.json    {"K": 3x3, "fps": int}
Untested on real footage until HUMAN_TODO H5; tested on synthetic clips written in this format.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.chores import CHORES
from perception.pipeline import accept_clip, perceive_video
from phases.events import segment
from phases.program import ObservedDemo
from retarget.hand2gripper import Waypoint, pseudo_demo_pairs, retarget_pick
from sim.scene import FIXTURES


def write_clip(path: Path, rgb: np.ndarray, K: np.ndarray, fps: int, depth: np.ndarray | None = None) -> Path:
    import imageio.v2 as imageio

    path.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path / "rgb.mp4", fps=fps, macro_block_size=1, quality=9) as w:
        for f in rgb:
            w.append_data(f)  # type: ignore[attr-defined]
    if depth is not None:
        np.savez_compressed(path / "depth.npz", depth=depth.astype(np.float32))
    (path / "intrinsics.json").write_text(json.dumps({"K": np.asarray(K).tolist(), "fps": int(fps)}))
    return path


def load_clip(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int]:
    import imageio.v2 as imageio

    meta = json.loads((path / "intrinsics.json").read_text())
    rgb = np.stack(list(imageio.mimread(path / "rgb.mp4", memtest=False)))
    depth = np.load(path / "depth.npz")["depth"] if (path / "depth.npz").exists() else None
    return rgb, depth, np.array(meta["K"]), int(meta["fps"])


def process_clip(chore_name: str, path: Path) -> dict:
    """Returns {"accepted", "reasons", "pairs": [(obj, obs, act)], "observed": ObservedDemo, "duration_s"}."""
    chore = CHORES[chore_name]
    objs = chore.object_specs()
    by_name = {o.name: o for o in objs}
    rgb, depth, K, fps = load_clip(path)
    vp = perceive_video(rgb, depth, K, fps, objs, rgb_only=depth is None)
    ok, why = accept_clip(vp)
    seg = segment(vp)
    pairs, picks_obs, at_place = [], [], []
    cur: Waypoint | None = None
    for pk in seg.picks:
        o = by_name[pk.obj]
        rs = retarget_pick(seg, pk, o, fps, mode="full", start=cur)
        base = pk.place_pos.copy()
        base[2] = max(0.0, pk.place_pos[2] - o.height / 2)
        if rs.accepted:
            pairs.append((pk.obj, *pseudo_demo_pairs(rs, o, base)))
            cur = rs.waypoints[-1]
        picks_obs.append((pk.obj, pk.pick_pos.copy(), base))
        snap = {}
        for n, tr in vp.objects.items():
            if n == pk.obj:
                continue
            prev = [q for q in seg.picks if q.obj == n and q.t_release < pk.t_grasp]
            okn = ~np.isnan(tr.pos[:, 0])
            if prev:
                b = prev[-1].place_pos.copy()
            elif okn.any():
                b = np.median(tr.pos[np.nonzero(okn)[0][: max(5, int(0.7 * fps))]], axis=0)
            else:
                continue
            b[2] -= by_name[n].height / 2
            snap[n] = b
        at_place.append(snap)
    fixtures = {f: np.array([FIXTURES[f].pos[0], FIXTURES[f].pos[1], FIXTURES[f].surface_z]) for f in chore.fixtures}
    if not seg.picks:
        ok, why = False, why + ["no pick-place detected"]
    return {"accepted": ok, "reasons": why, "pairs": pairs, "observed": ObservedDemo(picks_obs, at_place, fixtures),
            "duration_s": len(rgb) / fps, "quality": vp.quality}
