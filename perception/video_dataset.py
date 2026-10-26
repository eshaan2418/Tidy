"""Phone videos -> processed human demos (tracks, events, phases, retargeted pseudo-demos) + M3 metrics.

Sim stage: videos are synthesised by sim/human_demo.py (B-003). Real stage: replace `_load_video`
with a reader for data/private/videos/<chore>/clip_XX (RGB-D) and drop the GT-based metrics.
Processed outputs are cached under data/cache/ (git-ignored); manifests with hashes go to data/human/.
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.provenance import REPO_ROOT
from eval.chores import CHORES
from eval.placements import PlacementCard
from perception.pipeline import (
    HandEstimatorNoise,
    accept_clip,
    apply_hand_estimates,
    perceive_video,
    simulated_hand_estimator,
)
from phases.events import boundary_errors, event_f1, segment
from phases.program import ObservedDemo
from retarget.hand2gripper import RetargetedSegment, Waypoint, pseudo_demo_pairs, retarget_pick
from sim.scene import FIXTURES

CACHE = REPO_ROOT / "data" / "cache" / "human"
FILM_OVERHEAD_S = 6.0   # start/stop recording, framing (assumption; replaced by logged real minutes)
RESET_S = 20.0          # re-placing objects from a placement card (assumption, same for every method)


@dataclass
class DemoRecord:
    chore: str
    card: dict
    demo_seed: int
    variant: str
    accepted: bool
    reject_reasons: list[str]
    video_duration_s: float
    human_time_s: float
    observed: ObservedDemo | None
    segments: list[RetargetedSegment]
    pairs: list[tuple[str, np.ndarray, np.ndarray]]     # (obj, obs, act) per retargeted segment
    metrics: dict = field(default_factory=dict)
    key_frame: np.ndarray | None = None                  # small RGB frame for demo-request UIs/figures


def _cache_path(chore: str, card_seed: int, demo_seed: int, variant: str) -> Path:
    return CACHE / chore / f"{card_seed}_{demo_seed}_{variant}.pkl"


def process_demo(chore_name: str, card: PlacementCard, demo_seed: int, variant: str = "full",
                 hand_source: str = "sim_estimator", noise: HandEstimatorNoise | None = None,
                 rgb_only: bool = False, use_cache: bool = True) -> DemoRecord:
    """variant: full | raw (no retargeting). hand_source: sim_estimator | caps."""
    tag = f"{variant}_{hand_source}" + ("_rgb" if rgb_only else "") + (
        f"_n{noise.sigma_xy:.3f}_{noise.sigma_z:.3f}" if noise else "")
    path = _cache_path(chore_name, card.seed, demo_seed, tag)
    if use_cache and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    from sim.human_demo import record_demo

    chore = CHORES[chore_name]
    objs = chore.object_specs(card.subs)
    by_name = {o.name: o for o in objs}
    video = record_demo(chore, card, seed=demo_seed, with_depth=not rgb_only)
    vp = perceive_video(video.rgb, None if rgb_only else video.depth, video.K, video.fps, objs, rgb_only=rgb_only)
    if hand_source == "sim_estimator":
        vp = apply_hand_estimates(vp, simulated_hand_estimator(video.gt.hand, video.rgb, vp.cam_R, vp.cam_t,
                                                               noise or HandEstimatorNoise(), seed=demo_seed))
    ok, why = accept_clip(vp)
    seg = segment(vp)
    # ---- M3 metrics vs simulator GT (sim stage only)
    gt = video.gt
    ev = event_f1(seg.events, gt.events)
    be = boundary_errors(seg, gt.events)
    pick_err, place_err = [], []
    for pk in seg.picks:
        gi = int(pk.t_grasp * video.fps)
        gt_pick = gt.objects[pk.obj][max(0, gi - int(0.5 * video.fps))]
        pick_err.append(float(np.linalg.norm(pk.pick_pos[:2] - gt_pick[:2])))
        place_err.append(float(np.linalg.norm(pk.place_pos[:2] - gt.objects[pk.obj][-1][:2])))
    kp_err = {}
    for k in ("thumb", "index"):
        e = np.linalg.norm(getattr(vp, k) - gt.hand[k], axis=1)
        kp_err[k] = float(np.nanmedian(e)) if np.any(~np.isnan(e)) else None
    moved_gt = {e["obj"] for e in gt.events}
    moved_pred = {p.obj for p in seg.picks}
    metrics = {"event": ev, "boundary_err_s": be, "pick_err_m": pick_err, "place_err_m": place_err,
               "keypoint_median_err_m": kp_err, "quality": vp.quality,
               "moved_objects_correct": moved_gt == moved_pred}
    # ---- retargeting + pseudo-demos
    segments: list[RetargetedSegment] = []
    pairs = []
    cur: Waypoint | None = None
    picks_obs = []
    at_place = []
    for pk in seg.picks:
        o = by_name[pk.obj]
        rs = retarget_pick(seg, pk, o, video.fps, mode=variant, start=cur)
        segments.append(rs)
        place_base = pk.place_pos.copy()
        place_base[2] = max(0.0, pk.place_pos[2] - o.height / 2)
        if rs.accepted:
            obs, act = pseudo_demo_pairs(rs, o, place_base)
            pairs.append((pk.obj, obs, act))
            cur = rs.waypoints[-1]
        picks_obs.append((pk.obj, pk.pick_pos.copy(), place_base))
        # where every other object is when this one is placed: its settled place position if it was carried
        # earlier, else its rest position (robust to the hand occluding it during this place motion)
        snap = {}
        for n, tr in vp.objects.items():
            if n == pk.obj:
                continue
            prev = [q for q in seg.picks if q.obj == n and q.t_release < pk.t_grasp]
            if prev:
                b = prev[-1].place_pos.copy()
            else:
                ok_n = ~np.isnan(tr.pos[:, 0])
                if not ok_n.any():
                    continue
                b = np.median(tr.pos[np.nonzero(ok_n)[0][: max(5, int(0.7 * video.fps))]], axis=0)
            b[2] -= by_name[n].height / 2
            snap[n] = b
        at_place.append(snap)
    fixtures = {f: np.array([FIXTURES[f].pos[0], FIXTURES[f].pos[1], FIXTURES[f].surface_z]) for f in chore.fixtures}
    observed = ObservedDemo(picks_obs, at_place, fixtures)
    if not seg.picks:
        ok, why = False, why + ["no pick-place detected"]
    rec = DemoRecord(chore_name, card.to_json(), demo_seed, tag, ok, why, video.duration_s,
                     video.duration_s + FILM_OVERHEAD_S + RESET_S, observed, segments, pairs, metrics,
                     key_frame=video.rgb[int(len(video.rgb) * 0.1)][::2, ::2].copy())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fw:
        pickle.dump(rec, fw)
    return rec


def _job(args: tuple) -> DemoRecord:
    return process_demo(*args)


def process_many(jobs: list[tuple], workers: int | None = None) -> list[DemoRecord]:
    """jobs: tuples of process_demo positional args. Uses a process pool (MuJoCo rendering is per-process)."""
    todo = [j for j in jobs]
    workers = workers or max(1, min(len(todo), (os.cpu_count() or 4) - 2))
    if workers == 1:
        return [_job(j) for j in todo]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, todo))
