"""Phone video -> per-frame camera pose, 3D hand keypoints, object tracks (table frame).

Sim-stage stand-ins (B-003), each replaceable by the real model without changing the interface:
  * camera pose: 4 dark corner markers on the mat + PnP          (real: ArUco/ChArUco on the mat)
  * hand keypoints: coloured fingertip caps + skin blob + depth   (real: HaMeR/WiLoR-class 3D hand pose)
  * object masks: per-class colour segmentation + instance tracking (real: SAM-2-class masks + tracking)
  * depth: phone LiDAR-like depth; RGB-only fallback uses the table-plane / size priors
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from sim.scene import HUES, INDEX_HUE, MARKER_POS, THUMB_HUE, ObjectSpec

MARKER_WORLD = np.array([[x, y, 0.0015] for x, y in MARKER_POS])  # order: near-right, near-left, far-left, far-right
CAP_RADIUS = {"thumb": 0.009, "index": 0.008}


@dataclass
class Track:
    pos: np.ndarray          # (T,3), NaN where unknown
    yaw: np.ndarray          # (T,)
    visible: np.ndarray      # (T,) bool


@dataclass
class VideoPerception:
    T: int
    fps: int
    cam_R: np.ndarray        # (T,3,3) world->cam
    cam_t: np.ndarray        # (T,3)
    pnp_ok: np.ndarray       # (T,)
    thumb: np.ndarray        # (T,3) NaN if missing
    index: np.ndarray
    wrist: np.ndarray
    objects: dict[str, Track]
    quality: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------ helpers
def hsv_image(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def hue_mask(hsv: np.ndarray, hue_deg: float, tol_deg: float = 12, s_min: int = 120, v_min: int = 60) -> np.ndarray:
    h = hsv[..., 0].astype(np.int32) * 2  # OpenCV hue is 0..179
    d = np.abs(((h - hue_deg + 180) % 360) - 180)
    return (d <= tol_deg) & (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)


def backproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def cam_to_world(pc: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R.T @ (pc - t).T).T if pc.ndim == 2 else R.T @ (pc - t)


def ray_plane(u: float, v: float, K: np.ndarray, R: np.ndarray, t: np.ndarray, z_plane: float) -> np.ndarray:
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    o = -R.T @ t
    d = R.T @ d_cam
    s = (z_plane - o[2]) / d[2]
    return o + s * d


# ------------------------------------------------------------------------------------------ camera pose
def detect_marker_candidates(rgb: np.ndarray) -> list[tuple[float, float, float]]:
    hsv = hsv_image(rgb)
    dark = ((hsv[..., 2] < 28) & (hsv[..., 1] < 90)).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 4 <= area <= 600 and 0.3 < w / max(h, 1) < 3.5 and area / max(w * h, 1) > 0.45:
            out.append((float(cents[i][0]), float(cents[i][1]), float(area)))
    return out


def solve_camera(rgb: np.ndarray, K: np.ndarray, prior: tuple[np.ndarray, np.ndarray] | None = None):
    cands = detect_marker_candidates(rgb)
    if len(cands) < 4:
        return None
    cands = sorted(cands, key=lambda c: -c[2])[:9]
    best = None
    for combo in itertools.combinations(cands, 4):
        pts = np.array([[c[0], c[1]] for c in combo])
        # assign by image layout (phone on the human side, upright; see filming guide):
        # bottom row (larger v) = far-x corners; within a row, smaller u = -y
        order = np.argsort(pts[:, 1])
        top, bot = pts[order[:2]], pts[order[2:]]
        top = top[np.argsort(top[:, 0])]
        bot = bot[np.argsort(bot[:, 0])]
        img = np.array([top[1], top[0], bot[0], bot[1]], dtype=np.float64)
        # world order: (near,-y)? MARKER_POS = near-right(-y), near-left(+y), far-left(+y), far-right(-y)
        img_sorted = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float64)
        for cand_img in (img_sorted, img):
            ok, rvec, tvec = cv2.solvePnP(MARKER_WORLD, cand_img, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
                continue
            proj, _ = cv2.projectPoints(MARKER_WORLD, rvec, tvec, K, None)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - cand_img, axis=1)))
            R, _ = cv2.Rodrigues(rvec)
            cam_pos = -R.T @ tvec.ravel()
            if cam_pos[2] < 0.15 or cam_pos[0] < 0.25:  # must be above the table, on the human side
                continue
            if best is None or err < best[0]:
                best = (err, R, tvec.ravel())
    if best is None or best[0] > 3.0:
        return None
    return best[1], best[2], best[0]


def track_camera(rgb: np.ndarray, K: np.ndarray, prev: tuple[np.ndarray, np.ndarray],
                 max_px: float = 12.0) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Pose update from >= 3 markers matched to the projection of the previous pose (handles a marker
    occluded by an object or the hand; the ArUco version gets correspondences from marker IDs instead)."""
    cands = detect_marker_candidates(rgb)
    if len(cands) < 3:
        return None
    R0, t0 = prev
    rvec0, _ = cv2.Rodrigues(R0)
    proj, _ = cv2.projectPoints(MARKER_WORLD, rvec0, t0, K, None)
    proj = proj.reshape(-1, 2)
    c = np.array([[a, b] for a, b, _ in cands])
    world, img = [], []
    for j, pj in enumerate(proj):
        d = np.linalg.norm(c - pj, axis=1)
        k = int(np.argmin(d))
        if d[k] < max_px:
            world.append(MARKER_WORLD[j])
            img.append(c[k])
    if len(world) < 3:
        return None
    W, I = np.array(world), np.array(img, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(W, I, K, None, rvec0.copy(), t0.reshape(3, 1).copy(), useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    pr, _ = cv2.projectPoints(W, rvec, tvec, K, None)
    err = float(np.mean(np.linalg.norm(pr.reshape(-1, 2) - I, axis=1)))
    R, _ = cv2.Rodrigues(rvec)
    # with 3 points the pose is weakly constrained: reject jumps away from the previous pose
    if err > 3.0 or np.linalg.norm(tvec.ravel() - t0) > 0.03:
        return None
    return R, tvec.ravel(), err


def solve_camera_sequence(rgb: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Per-frame camera pose: full 4-marker solves where possible, then forward/backward tracking with >= 3
    markers from the nearest solved frame, then hold-last for the remaining gaps (marked not ok)."""
    T = len(rgb)
    Rs = np.full((T, 3, 3), np.nan)
    ts = np.full((T, 3), np.nan)
    ok = np.zeros(T, bool)
    errs: list[float] = []
    for i in range(T):
        sol = solve_camera(rgb[i], K)
        if sol is not None:
            Rs[i], ts[i], ok[i] = sol[0], sol[1], True
            errs.append(sol[2])
    if not ok.any():
        return Rs, ts, ok, errs
    for order in (range(T), range(T - 1, -1, -1)):
        prev = None
        for i in order:
            if ok[i]:
                prev = (Rs[i], ts[i])
                continue
            if prev is None:
                continue
            sol = track_camera(rgb[i], K, prev)
            if sol is not None:
                Rs[i], ts[i], ok[i] = sol[0], sol[1], True
                errs.append(sol[2])
                prev = (Rs[i], ts[i])
    return Rs, ts, ok, errs


# ------------------------------------------------------------------------------------------ keypoints
def blob_center(mask: np.ndarray, min_px: int = 3) -> tuple[float, float, int] | None:
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_px:
        return None
    return float(cents[i][0]), float(cents[i][1]), int(stats[i, cv2.CC_STAT_AREA])


def point_from_mask(mask: np.ndarray, depth: np.ndarray | None, K, R, t, radius: float,
                    rgb_only_size_prior: bool) -> np.ndarray | None:
    bc = blob_center(mask)
    if bc is None:
        return None
    u, v, area = bc
    if depth is not None and not rgb_only_size_prior:
        ds = depth[mask]
        ds = ds[(ds > 0.05) & (ds < 2.0)]
        if len(ds) == 0:
            return None
        z = float(np.median(ds)) + radius * 0.6  # surface -> centre along the ray (approx.)
    else:
        r_px = np.sqrt(area / np.pi)
        z = float(K[0, 0] * radius / max(r_px, 0.5))
    return cam_to_world(backproject(u, v, z, K), R, t)


def skin_mask_frame(hsv: np.ndarray) -> np.ndarray:
    h = hsv[..., 0].astype(np.int32) * 2
    return (h >= 5) & (h <= 40) & (hsv[..., 1] >= 60) & (hsv[..., 1] <= 170) & (hsv[..., 2] >= 90)


# ------------------------------------------------------------------------------------------ objects
def object_points(mask: np.ndarray, depth: np.ndarray | None, K, R, t, obj: ObjectSpec,
                  rgb_only: bool) -> np.ndarray | None:
    vs, us = np.nonzero(mask)
    if len(us) < 6:
        return None
    if depth is not None and not rgb_only:
        z = depth[vs, us]
        keep = (z > 0.05) & (z < 2.0)
        if keep.sum() < 6:
            return None
        us, vs, z = us[keep], vs[keep], z[keep]
        pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], axis=1)
        return cam_to_world(pc, R, t)
    # RGB-only: intersect rays with the object's top-surface plane (object resting on the table)
    zt = obj.height
    pts = np.array([ray_plane(float(u), float(v), K, R, t, zt) for u, v in zip(us[::3], vs[::3])])
    return pts


def object_estimate(pw: np.ndarray, obj: ObjectSpec) -> tuple[np.ndarray, float]:
    # use the upper part of the visible surface to estimate the footprint centre
    ztop = np.percentile(pw[:, 2], 90)
    top = pw[pw[:, 2] > ztop - 0.012]
    if len(top) < 4:
        top = pw
    xy = top[:, :2].mean(axis=0)
    c = top[:, :2] - xy
    yaw = 0.0
    if len(c) >= 5:
        w, V = np.linalg.eigh(c.T @ c)
        major = V[:, 1]
        yaw = float(np.arctan2(major[1], major[0]))
    zc = max(float(np.percentile(pw[:, 2], 95)) - obj.height / 2, obj.height / 2)
    return np.array([xy[0], xy[1], zc]), yaw


# ------------------------------------------------------------------------------------------ main entry
def perceive_video(rgb: np.ndarray, depth: np.ndarray | None, K: np.ndarray, fps: int,
                   objects: list[ObjectSpec], rgb_only: bool = False) -> VideoPerception:
    T = len(rgb)
    cam_R = np.full((T, 3, 3), np.nan)
    cam_t = np.full((T, 3), np.nan)
    pnp_ok = np.zeros(T, bool)
    thumb = np.full((T, 3), np.nan)
    index = np.full((T, 3), np.nan)
    wrist = np.full((T, 3), np.nan)
    tracks = {o.name: Track(np.full((T, 3), np.nan), np.full(T, np.nan), np.zeros(T, bool)) for o in objects}
    sol_R, sol_t, sol_ok, reproj = solve_camera_sequence(rgb, K)
    last_pose = None
    by_color: dict[str, list[ObjectSpec]] = {}
    for o in objects:
        by_color.setdefault(o.color, []).append(o)
    last_xy: dict[str, np.ndarray] = {}
    first = int(np.argmax(sol_ok)) if sol_ok.any() else T
    for i in range(T):
        im = rgb[i]
        if sol_ok[i]:
            pnp_ok[i] = True
            last_pose = (sol_R[i], sol_t[i])
        elif last_pose is None and i < first and sol_ok.any():
            last_pose = (sol_R[first], sol_t[first])  # leading frames: nearest solved pose
        if last_pose is None:
            continue
        R, tv = last_pose
        cam_R[i], cam_t[i] = R, tv
        hsv = hsv_image(im)
        d = None if depth is None else depth[i]
        for key, hue, arr in (("thumb", THUMB_HUE, thumb), ("index", INDEX_HUE, index)):
            m = hue_mask(hsv, hue, 14, 90, 60)
            p = point_from_mask(m, d, K, R, tv, CAP_RADIUS[key], rgb_only)
            if p is not None:
                arr[i] = p
        sk = skin_mask_frame(hsv)
        if sk.sum() > 30 and not np.isnan(thumb[i, 0]):
            vs, us = np.nonzero(sk)
            if d is not None and not rgb_only:
                z = d[vs, us]
                keep = (z > 0.05) & (z < 2.0)
                pc = np.stack([(us[keep] - K[0, 2]) * z[keep] / K[0, 0], (vs[keep] - K[1, 2]) * z[keep] / K[1, 1],
                               z[keep]], axis=1)
                pw = cam_to_world(pc, R, tv)
                mid = 0.5 * (thumb[i] + (index[i] if not np.isnan(index[i, 0]) else thumb[i]))
                near = pw[np.linalg.norm(pw - mid, axis=1) < 0.11]
                if len(near) > 10:
                    wrist[i] = near.mean(axis=0)
        for color, objs in by_color.items():
            m = hue_mask(hsv, HUES[color], 12, 120, 50)
            m &= ~sk
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            comps = [j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= 8]
            ests = []
            for j in comps:
                pts = object_points(lab == j, d, K, R, tv, objs[0], rgb_only)
                if pts is None:
                    continue
                pos, yaw = object_estimate(pts, objs[0])
                ests.append((pos, yaw, stats[j, cv2.CC_STAT_AREA]))
            ests.sort(key=lambda e: -e[2])
            ests = ests[:len(objs)]
            # instance assignment: greedy nearest to last known position
            unassigned = list(objs)
            for pos, yaw, _ in ests:
                if not unassigned:
                    break
                if all(o.name in last_xy for o in unassigned):
                    o = min(unassigned, key=lambda o: float(np.linalg.norm(last_xy[o.name] - pos[:2])))
                else:
                    o = next(o for o in unassigned if o.name not in last_xy) if any(
                        o.name not in last_xy for o in unassigned) else unassigned[0]
                    # first sighting: order instances by y for determinism
                unassigned.remove(o)
                tr = tracks[o.name]
                tr.pos[i], tr.yaw[i], tr.visible[i] = pos, yaw, True
                last_xy[o.name] = pos[:2]
    quality = {"pnp_rate": float(pnp_ok.mean()), "pnp_reproj_px": float(np.mean(reproj)) if reproj else None,
               "thumb_rate": float(np.mean(~np.isnan(thumb[:, 0]))),
               "index_rate": float(np.mean(~np.isnan(index[:, 0]))),
               "obj_rate": {n: float(tr.visible.mean()) for n, tr in tracks.items()}}
    return VideoPerception(T, fps, cam_R, cam_t, pnp_ok, thumb, index, wrist, tracks, quality)


def accept_clip(vp: VideoPerception) -> tuple[bool, list[str]]:
    """Automatic rejection of unusable clips (M3 quality checks)."""
    why = []
    q = vp.quality
    if q["pnp_rate"] < 0.6:
        why.append(f"mat markers visible in only {100*q['pnp_rate']:.0f}% of frames")
    if q["pnp_reproj_px"] is not None and q["pnp_reproj_px"] > 2.0:
        why.append("marker reprojection error > 2 px")
    if max(q["thumb_rate"], q["index_rate"]) < 0.4:
        why.append("no fingertip visible in >= 40% of frames")
    for n, r in q["obj_rate"].items():
        if r < 0.2:
            why.append(f"object {n} visible in < 20% of frames")
    return (not why), why


# ------------------------------------------------------------------------------------------ simulated estimator
@dataclass
class HandEstimatorNoise:
    """Error model for the SIMULATED HaMeR/WiLoR-class 3D hand estimator (sim stage only, D-013).
    Camera-frame Gaussian noise, AR(1)-correlated over time; depth (z) error larger than image-plane error."""
    sigma_xy: float = 0.006
    sigma_z: float = 0.012
    rho: float = 0.8
    min_skin_px: int = 30


def simulated_hand_estimator(gt_hand: dict[str, np.ndarray], rgb: np.ndarray, cam_R: np.ndarray, cam_t: np.ndarray,
                             noise: HandEstimatorNoise, seed: int = 0) -> dict[str, np.ndarray]:
    """Stand-in for a learned 3D hand-pose model: GT keypoints + calibrated noise, output only on frames where
    the hand is visible in the image (skin pixels) and the camera pose is known. Labelled `sim_estimator`."""
    rng = np.random.default_rng(seed)
    T = len(rgb)
    out = {k: np.full((T, 3), np.nan) for k in ("thumb", "index", "wrist")}
    e = {k: np.zeros(3) for k in out}
    sig = np.array([noise.sigma_xy, noise.sigma_xy, noise.sigma_z])
    for i in range(T):
        for k in out:
            e[k] = noise.rho * e[k] + np.sqrt(1 - noise.rho**2) * rng.normal(0, 1, 3) * sig
        if np.isnan(cam_R[i, 0, 0]):
            continue
        if skin_mask_frame(hsv_image(rgb[i])).sum() < noise.min_skin_px:
            continue
        R, t = cam_R[i], cam_t[i]
        common = rng.normal(0, 1, 3) * sig * 0.3
        for k in out:
            pc = R @ gt_hand[k][i] + t
            out[k][i] = R.T @ (pc + e[k] + common - t)
    return out


def apply_hand_estimates(vp: VideoPerception, est: dict[str, np.ndarray]) -> VideoPerception:
    vp.thumb, vp.index, vp.wrist = est["thumb"], est["index"], est["wrist"]
    vp.quality["hand_source"] = "sim_estimator"
    vp.quality["thumb_rate"] = float(np.mean(~np.isnan(vp.thumb[:, 0])))
    vp.quality["index_rate"] = float(np.mean(~np.isnan(vp.index[:, 0])))
    return vp
