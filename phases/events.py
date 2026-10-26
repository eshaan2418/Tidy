"""Contact / grasp-release events, task-phase segmentation and observed pick-place program from video tracks.

Unsupervised and rule-based (no labels needed): an object's *movement onset* and *settling* define a
carry interval; the hand aperture (thumb-index distance) refines grasp/release times.  Phases are then:
reach -> grasp -> transport -> place -> retreat for each carried object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from perception.pipeline import VideoPerception


@dataclass
class Event:
    type: str      # grasp | release
    t: float
    obj: str


@dataclass
class PickPlace:
    obj: str
    t_grasp: float
    t_release: float
    pick_pos: np.ndarray       # object rest position before the grasp (table frame)
    pick_yaw: float
    place_pos: np.ndarray      # object position after it settles
    t_segment_start: float     # start of this object's reach


@dataclass
class Segmentation:
    events: list[Event]
    phases: list[tuple[str, float, float, str]]
    picks: list[PickPlace]
    thumb: np.ndarray          # gap-filled keypoints (T,3)
    index: np.ndarray
    wrist: np.ndarray


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN gaps up to max_gap frames (per column)."""
    x = x.copy()
    T = len(x)
    valid = ~np.isnan(x[:, 0])
    if valid.sum() < 2:
        return x
    idx = np.arange(T)
    for c in range(x.shape[1]):
        xi = np.interp(idx, idx[valid], x[valid, c])
        x[:, c] = np.where(valid, x[:, c], xi)
    # re-blank gaps longer than max_gap
    run = 0
    for i in range(T):
        if not valid[i]:
            run += 1
        else:
            if run > max_gap:
                x[i - run:i] = np.nan
            run = 0
    if run > 0:
        x[T - run:] = np.nan  # trailing gap: unknown
    lead = np.argmax(valid)
    x[:lead] = np.nan
    return x


def complete_hand(vp: VideoPerception) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hand prior: fill short gaps; propagate the thumb->index offset when one tip is occluded."""
    g = int(0.5 * vp.fps)
    th, ix, wr = fill_gaps(vp.thumb, g), fill_gaps(vp.index, g), fill_gaps(vp.wrist, g)
    off = None
    for i in range(vp.T):
        if not np.isnan(th[i, 0]) and not np.isnan(ix[i, 0]):
            off = ix[i] - th[i]
        elif off is not None and not np.isnan(th[i, 0]):
            ix[i] = th[i] + off
        elif off is not None and not np.isnan(ix[i, 0]):
            th[i] = ix[i] - off
    return th, ix, wr


def smooth_track(p: np.ndarray, fps: int) -> np.ndarray:
    q = fill_gaps(p, int(1.0 * fps))
    ok = ~np.isnan(q[:, 0])
    if ok.sum() > 5:
        q[ok] = median_filter(q[ok], size=(5, 1), mode="nearest")
    return q


def uniform_filter1d_safe(x: np.ndarray, n: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    y = np.nan_to_num(x, nan=0.0)
    return uniform_filter1d(y, n, mode="nearest")


def segment(vp: VideoPerception, move_thresh: float = 0.04) -> Segmentation:
    fps = vp.fps
    th, ix, wr = complete_hand(vp)
    aperture = np.linalg.norm(th - ix, axis=1)
    ok_a = ~np.isnan(aperture)
    if ok_a.sum() > 9:  # keypoint noise -> smooth the aperture before thresholding
        from scipy.ndimage import uniform_filter1d

        aperture[ok_a] = uniform_filter1d(median_filter(aperture[ok_a], size=7, mode="nearest"), 9, mode="nearest")
    gc_all = 0.5 * (th + ix)
    speed = np.r_[0.0, np.linalg.norm(np.diff(gc_all[:, :2], axis=0), axis=1) * fps]
    if np.any(~np.isnan(speed)):
        speed = np.where(np.isnan(speed), np.nan, uniform_filter1d_safe(speed, 7))
    picks: list[PickPlace] = []
    for name, tr in vp.objects.items():
        p = smooth_track(tr.pos, fps)
        ok = ~np.isnan(p[:, 0])
        if ok.sum() < 10:
            continue
        first = np.nonzero(ok)[0][: max(5, int(0.7 * fps))]
        rest = np.median(p[first], axis=0)
        yaw_rest = float(np.nanmedian(tr.yaw[first])) if np.any(~np.isnan(tr.yaw[first])) else 0.0
        d = np.linalg.norm(p[:, :2] - rest[:2], axis=1)
        # onset: earliest frame after which the object never returns to its rest position (unknown frames
        # are ignored), with the hand's grasp centre near the rest position shortly before -> robust to
        # transient occlusion of the object by the hand passing over it
        far = np.where(np.isnan(d), True, d > move_thresh)
        suffix = np.logical_and.accumulate(far[::-1])[::-1]
        grasp_c = 0.5 * (th + ix)
        hand_d = np.linalg.norm(grasp_c[:, :2] - rest[:2], axis=1)
        onset: int | None = None
        for i in np.nonzero(suffix & ok)[0]:
            w = hand_d[max(0, i - int(1.0 * fps)):i + int(0.3 * fps)]
            if np.any(np.nan_to_num(w, nan=1.0) < 0.045):
                onset = int(i)
                break
        if onset is None or onset > vp.T - 3:
            continue  # never (durably) moved: static object
        final = np.nanmedian(p[-int(0.5 * fps):], axis=0) if np.any(ok[-int(0.5 * fps):]) else p[ok][-1]
        df = np.linalg.norm(p[:, :2] - final[:2], axis=1)
        settle: int | None = None
        for kk in range(onset, vp.T):
            stays = np.all(np.nan_to_num(df[kk:kk + int(0.3 * fps)], nan=0) < 0.02)
            if not np.isnan(df[kk]) and df[kk] < 0.012 and stays:
                settle = kk
                break
        if settle is None:
            settle = int(vp.T - 1)
        # refine with the aperture
        w0 = max(0, onset - int(1.2 * fps))
        a_win = aperture[w0:onset + 1]
        if np.any(~np.isnan(a_win)):
            amin = np.nanmin(a_win)
            closed = np.nonzero(np.nan_to_num(a_win, nan=1.0) <= amin + 0.006)[0]
            i_g = w0 + int(closed[0]) if len(closed) else onset
        else:
            i_g = max(0, onset - int(0.15 * fps))
        a_g = aperture[i_g] if not np.isnan(aperture[i_g]) else np.nan
        # release: the hand lowers the object, stops at the bottom of the place motion and opens. Release =
        # first frame where the smoothed grasp-centre height reaches the bottom of that motion, searched near
        # the place location and before the object settles for good; aperture opening is the fallback cue.
        gz = uniform_filter1d_safe(gc_all[:, 2], 7)
        gz = np.where(np.isnan(gc_all[:, 2]), np.nan, gz)
        near_final = np.linalg.norm(grasp_c[:, :2] - final[:2], axis=1) < 0.05
        w_lo, w_hi = onset + int(0.3 * fps), min(vp.T, settle + int(1.0 * fps))
        win = np.arange(w_lo, w_hi)
        win = win[near_final[win] & ~np.isnan(gz[win])] if len(win) else win
        i_r = settle
        if len(win) >= 3:
            zmin = float(np.min(gz[win]))
            i_r = int(win[np.nonzero(gz[win] <= zmin + 0.006)[0][0]])
        elif not np.isnan(a_g):
            opened = np.nan_to_num(aperture, nan=0.0) > a_g + 0.012
            cand = np.nonzero(opened[onset:])[0]
            if len(cand):
                i_r = onset + int(cand[0])
        settle = max(settle, i_r)
        after = p[min(vp.T - 1, i_r + int(0.4 * fps)):min(vp.T, i_r + int(1.0 * fps))]
        place = np.nanmedian(after, axis=0) if np.any(~np.isnan(after[:, 0])) else final
        picks.append(PickPlace(name, i_g / fps, i_r / fps, rest, yaw_rest, place, 0.0))
    picks.sort(key=lambda k: k.t_grasp)
    events: list[Event] = []
    phases: list[tuple[str, float, float, str]] = []
    prev_end = 0.0
    for k in picks:
        k.t_segment_start = prev_end
        events += [Event("grasp", k.t_grasp, k.obj), Event("release", k.t_release, k.obj)]
        t_place = max(k.t_grasp, k.t_release - 0.6)
        phases += [("reach", prev_end, max(prev_end, k.t_grasp - 0.4), k.obj),
                   ("grasp", max(prev_end, k.t_grasp - 0.4), k.t_grasp, k.obj),
                   ("transport", k.t_grasp, t_place, k.obj), ("place", t_place, k.t_release, k.obj),
                   ("retreat", k.t_release, k.t_release + 0.5, k.obj)]
        prev_end = k.t_release + 0.5
    return Segmentation(events, phases, picks, th, ix, wr)


# ------------------------------------------------------------------------------------------ metrics vs GT
def instance_class(name: str) -> str:
    """Identical instances (block_red / block_red2) are interchangeable for the task and for scoring."""
    return name.rstrip("0123456789")


def event_f1(pred: list[Event], gt: list[dict], tol: float = 0.3) -> dict:
    used: set[int] = set()
    tp = 0
    errs: list[float] = []
    for e in pred:
        best: float | None = None
        bj: int | None = None
        for j, g in enumerate(gt):
            if j in used or g["type"] != e.type or instance_class(g["obj"]) != instance_class(e.obj):
                continue
            d = abs(g["t"] - e.t)
            if d <= tol and (best is None or d < best):
                best, bj = d, j
        if bj is not None and best is not None:
            used.add(bj)
            tp += 1
            errs.append(best)
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "mean_abs_time_err_s": float(np.mean(errs)) if errs else None}


def boundary_errors(pred: Segmentation, gt_events: list[dict]) -> list[float]:
    """|t_pred - t_gt| for every GT grasp/release that has a same-object prediction (unmatched -> excluded)."""
    out = []
    for g in gt_events:
        cands = [e.t for e in pred.events if e.type == g["type"] and instance_class(e.obj) == instance_class(g["obj"])]
        if cands:
            out.append(min(abs(c - g["t"]) for c in cands))
    return out
