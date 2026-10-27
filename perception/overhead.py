"""Robot-side perception from the fixed overhead RGB-D camera (used at runtime by every learned agent).

Per object step the executor asks for the pose of the target object (and landmark objects).  This module
segments the object by colour (sim stand-in for SAM-2-class masks + a class prompt), back-projects the
mask with depth using the calibrated camera (intrinsics + extrinsics from hardware/camera_calib.py on the
real robot; the model camera in sim), and fits position (top-surface centroid) + pi-symmetric yaw (PCA).

Identical instances (two red blocks) are assigned to remembered positions by minimum total distance.
If an object is not visible (e.g., occluded by the arm), the last estimate is returned and flagged.
`calib_offset` simulates a bumped camera (extrinsic error) for the M7 camera-bump test.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import cv2
import mujoco
import numpy as np

from common.geometry import wrap_half_pi
from perception.pipeline import hue_mask, size_plausible
from sim.scene import HUES

W, H = 320, 240


@dataclass
class CameraModel:
    K: np.ndarray
    R: np.ndarray      # world -> cam (OpenCV)
    t: np.ndarray

    @staticmethod
    def from_mujoco(model: mujoco.MjModel, data: mujoco.MjData, name: str, w: int = W, h: int = H) -> CameraModel:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        fovy = float(model.cam_fovy[cid])
        f = 0.5 * h / np.tan(np.deg2rad(fovy) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        Rc = data.cam_xmat[cid].reshape(3, 3)       # columns: cam x, y, z axes in world (MuJoCo: looks along -z)
        R = np.stack([Rc[:, 0], -Rc[:, 1], -Rc[:, 2]])
        return CameraModel(K, R, -R @ data.cam_xpos[cid])


@dataclass
class OverheadPerceiver:
    """Colour + depth perception from the overhead camera. Implements eval.runner.Perceiver."""
    calib_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world-frame extrinsic error (m)
    min_px: int = 6
    memory: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)
    hue_cal: dict[str, float] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: {"seen": 0, "missed": 0})
    _cam: CameraModel | None = None
    size_gate: Any = None   # ObjectSpec of the class being segmented (set per call in perceive)

    def reset(self) -> None:
        self.memory.clear()

    def calibrate(self, env) -> None:
        """Session-start colour calibration (D-018): the rendered hue of each colour class under the current
        lighting, from a reference image where each object's pixels are known (real robot: the operator
        clicks each object once in the setup UI; sim: MuJoCo segmentation render)."""
        r = mujoco.Renderer(env.model, H, W)
        r.update_scene(env.data, "overhead")
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(env.data, "overhead")
        seg = r.render().copy()
        r.close()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[..., 0].astype(np.float64) * 2
        for o in env.all_objects:
            bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{o.name}")
            gids = [g for g in range(env.model.ngeom) if env.model.geom_bodyid[g] == bid]
            m = np.isin(seg[..., 0], gids) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (hsv[..., 1] > 80)
            if m.sum() >= 5 and o.color not in self.hue_cal:
                a = np.deg2rad(hue[m])
                self.hue_cal[o.color] = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)

    def _camera(self, env) -> CameraModel:
        if self._cam is None:
            self._cam = CameraModel.from_mujoco(env.model, env.data, "overhead")
        return self._cam

    def _components(self, env, rgb: np.ndarray, depth: np.ndarray, color: str, height: float):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = hue_mask(hsv, self.hue_cal.get(color, HUES[color]), 8, 90, 35).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cam = self._camera(env)
        out = []
        for j in range(1, n):
            if st[j, cv2.CC_STAT_AREA] < self.min_px:
                continue
            vs, us = np.nonzero(lab == j)
            z = depth[vs, us]
            keep = (z > 0.05) & (z < 2.0)
            if keep.sum() < self.min_px:
                continue
            us, vs, z = us[keep], vs[keep], z[keep]
            pc = np.stack([(us - cam.K[0, 2]) * z / cam.K[0, 0], (vs - cam.K[1, 2]) * z / cam.K[1, 1], z], axis=1)
            pw = (cam.R.T @ (pc - cam.t).T).T + self.calib_offset
            if self.size_gate is not None and not size_plausible(pw, self.size_gate):
                continue
            ztop = float(np.percentile(pw[:, 2], 90))
            top = pw[pw[:, 2] > ztop - 0.006]
            if len(top) < 3:
                top = pw
            xy = top[:, :2].mean(axis=0)
            yaw = 0.0
            if len(top) >= 5:
                c = top[:, :2] - xy
                _, V = np.linalg.eigh(c.T @ c)
                yaw = float(np.arctan2(V[1, 1], V[0, 1]))
            pos = np.array([xy[0], xy[1], max(ztop - height / 2, height / 2)])
            out.append((pos, float(wrap_half_pi(yaw)), int(len(us))))
        return out

    def perceive(self, env, names: list[str]) -> dict[str, tuple[np.ndarray, float]]:
        if not self.hue_cal:
            self.calibrate(env)
        rgb, depth = env.render("overhead", W, H, depth=True)
        res: dict[str, tuple[np.ndarray, float]] = {}
        # group requested names with all same-colour, same-shape objects in the scene (instance assignment)
        groups: dict[tuple[str, str, tuple], list[str]] = {}
        for o in env.all_objects:
            groups.setdefault((o.color, o.kind, tuple(o.size)), []).append(o.name)
        done: set[str] = set()
        for n in names:
            if n in done:
                continue
            o = env.obj_by_name[n]
            members = groups[(o.color, o.kind, tuple(o.size))]
            self.size_gate = o
            comps = self._components(env, rgb, depth, o.color, o.height)
            self.size_gate = None
            # other-shaped objects of the same colour: drop components that better match them by size
            comps.sort(key=lambda c: -c[2])
            comps = comps[: max(len(members), 1) + 2]
            assign = self._assign(members, comps)
            for m in members:
                if m in assign:
                    self.memory[m] = assign[m]
                    self.stats["seen"] += 1
                elif m in names:
                    self.stats["missed"] += 1
                if m in self.memory:
                    res[m] = (self.memory[m][0].copy(), self.memory[m][1])
                done.add(m)
        for n in names:
            if n not in res:  # never seen: fall back to the table centre (the step will fail and be scored)
                res[n] = (np.array([0.18, 0.0, env.obj_by_name[n].height / 2]), 0.0)
        return res

    def _assign(self, members: list[str], comps: list) -> dict[str, tuple[np.ndarray, float]]:
        if not comps:
            return {}
        comps = comps[: len(members) + 2]
        known = [m for m in members if m in self.memory]
        if len(members) == 1:
            m = members[0]
            if m in self.memory:  # prefer the component near the remembered position, else the largest
                best = min(comps, key=lambda c: float(np.linalg.norm(c[0][:2] - self.memory[m][0][:2])) - 1e-4 * c[2])
                if np.linalg.norm(best[0][:2] - self.memory[m][0][:2]) > 0.25:
                    best = comps[0]
            else:
                best = comps[0]
            return {m: (best[0], best[1])}
        if not known:  # first sighting: deterministic order by y
            cs = sorted(comps[: len(members)], key=lambda c: c[0][1])
            return {m: (c[0], c[1]) for m, c in zip(members, cs)}
        best_cost, best_perm = None, None
        k = min(len(members), len(comps))
        for perm in itertools.permutations(range(len(comps)), k):
            cost = 0.0
            for m, ci in zip(members, perm):
                ref = self.memory.get(m)
                cost += 0.05 if ref is None else float(np.linalg.norm(comps[ci][0][:2] - ref[0][:2]))
            if best_cost is None or cost < best_cost:
                best_cost, best_perm = cost, perm
        assert best_perm is not None
        return {m: (comps[ci][0], comps[ci][1]) for m, ci in zip(members, best_perm)}


@dataclass
class RayPlanePerceiver(OverheadPerceiver):
    """Depth-free variant for the real robot's RGB overhead webcam: every colour component is back-projected by
    intersecting pixel rays with the plane at the object's known top height (class + size come from the object
    library). `camera` is the calibrated model (hardware/camera_calib.py); in sim it is the MuJoCo camera."""
    camera: CameraModel | None = None

    def _camera(self, env) -> CameraModel:
        if self.camera is None:
            self.camera = CameraModel.from_mujoco(env.model, env.data, "overhead")
        return self.camera

    def _components(self, env, rgb: np.ndarray, depth: np.ndarray | None, color: str, height: float):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = hue_mask(hsv, self.hue_cal.get(color, HUES[color]), 8, 90, 35).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cam = self._camera(env)
        out = []
        for j in range(1, n):
            if st[j, cv2.CC_STAT_AREA] < self.min_px:
                continue
            vs, us = np.nonzero(lab == j)
            rays = np.stack([(us - cam.K[0, 2]) / cam.K[0, 0], (vs - cam.K[1, 2]) / cam.K[1, 1], np.ones(len(us))], 1)
            d_w = (cam.R.T @ rays.T).T                    # ray directions in the world frame
            o_w = -cam.R.T @ cam.t                        # camera centre
            s = (height - o_w[2]) / d_w[:, 2]
            pw = o_w + s[:, None] * d_w + self.calib_offset
            if self.size_gate is not None and not size_plausible(pw, self.size_gate):
                continue
            xy = pw[:, :2].mean(axis=0)
            yaw = 0.0
            if len(pw) >= 5:
                c = pw[:, :2] - xy
                _, V = np.linalg.eigh(c.T @ c)
                yaw = float(np.arctan2(V[1, 1], V[0, 1]))
            out.append((np.array([xy[0], xy[1], height / 2]), float(wrap_half_pi(yaw)), int(len(us))))
        return out

    def perceive(self, env, names: list[str]) -> dict[str, tuple[np.ndarray, float]]:
        # identical to the RGB-D perceiver except that no depth image is requested
        orig = env.render

        def rgb_only(camera="overhead", width=W, height=H, depth=False):
            img = orig(camera, width, height)
            return (img, None) if depth else img

        env.render = rgb_only
        try:
            return super().perceive(env, names)
        finally:
            env.render = orig
