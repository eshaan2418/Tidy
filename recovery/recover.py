"""Per-phase failure detection and reset-and-retry recovery (Contribution 3).

Detector inputs are only what the real robot has after an object step: proprioception (jaw opening while the
gripper was commanded closed, lift ticks, timeout) and the overhead-camera pose of the object before/after the
step relative to the perceived goal.  Two calibrated logistic classifiers:
  * grasp phase: P(object was grasped and lifted)
  * place phase: P(object ended at its goal)
A step is flagged as failed if P(place ok) < threshold.  Its *phase* (grasp vs place) is the one with the lower
probability.  Recovery: safe retreat (open, rise vertically) -> re-perceive -> retry the failed object step
(max N retries) -> ask for help.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim.env import EEAction
from sim.expert import TRANSIT_Z, jaw_half_extent

FEATURES = ("closed_gap", "lifted", "timeout", "dist_goal_xy", "dz_goal", "moved_xy", "ticks", "obj_width")


def step_features(rec, obj) -> np.ndarray:
    width = 2 * jaw_half_extent(obj)
    closed = rec.grip_closed_open if rec.grip_closed_open is not None else 0.0
    fp = rec.final_perceived if rec.final_perceived is not None else rec.perceived_obj
    g = rec.goal if rec.goal is not None else fp
    d_goal = float(np.linalg.norm(fp[:2] - g[:2]))
    dz = float(fp[2] - obj.height / 2 - g[2])
    moved = float(np.linalg.norm(fp[:2] - rec.perceived_obj[:2])) if rec.perceived_obj is not None else 0.0
    return np.array([(closed - width) * 100, min(rec.lifted_ticks, 30) / 10, float(rec.timeout), d_goal * 100,
                     dz * 100, moved * 100, rec.ticks / 100, width * 100])


@dataclass
class Logistic:
    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray

    def proba(self, X: np.ndarray) -> np.ndarray:
        z = ((np.atleast_2d(X) - self.mu) / self.sd) @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 3000, lr: float = 0.1) -> Logistic:
        mu, sd = X.mean(0), np.maximum(X.std(0), 1e-3)
        Z = (X - mu) / sd
        w, b = np.zeros(X.shape[1]), 0.0
        pos = max(1, int(y.sum()))
        neg = max(1, int(len(y) - y.sum()))
        sw = np.where(y > 0.5, len(y) / (2 * pos), len(y) / (2 * neg))   # class-balanced
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = (p - y) * sw
            w -= lr * (Z.T @ g / len(y) + l2 * w)
            b -= lr * float(g.mean())
        return Logistic(w, b, mu, sd)

    def to_json(self) -> dict:
        return {"w": self.w.tolist(), "b": self.b, "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @staticmethod
    def from_json(d: dict) -> Logistic:
        return Logistic(np.array(d["w"]), float(d["b"]), np.array(d["mu"]), np.array(d["sd"]))


@dataclass
class FailureDetector:
    grasp: Logistic
    place: Logistic
    threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    @staticmethod
    def fit(X: np.ndarray, y_grasp: np.ndarray, y_place: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None, target_precision: float = 0.8) -> FailureDetector:
        det = FailureDetector(Logistic.fit(X, y_grasp), Logistic.fit(X, y_place))
        if X_val is not None and y_val is not None and len(X_val):
            det.threshold = det.calibrate(X_val, y_val, target_precision)
        return det

    def calibrate(self, X: np.ndarray, y_place: np.ndarray, target_precision: float) -> float:
        """Largest-recall threshold whose failure-detection precision on held-out steps is >= target."""
        p = self.place.proba(X)
        fail = y_place < 0.5
        best = 0.5
        best_rec = -1.0
        for th in np.linspace(0.05, 0.95, 91):
            flag = p < th
            if flag.sum() == 0:
                continue
            tp = int((flag & fail).sum())
            prec = tp / int(flag.sum())
            rec = tp / max(1, int(fail.sum()))
            if prec >= target_precision and rec > best_rec:
                best, best_rec = float(th), rec
        return best

    def judge(self, x: np.ndarray) -> tuple[bool, str, float]:
        pg = float(self.grasp.proba(x)[0])
        pp = float(self.place.proba(x)[0])
        failed = pp < self.threshold
        phase = "grasp" if pg < pp else "place"
        return failed, phase, pp

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"grasp": self.grasp.to_json(), "place": self.place.to_json(),
                                    "threshold": self.threshold, "features": FEATURES, "meta": self.meta}, indent=1))

    @staticmethod
    def load(path: Path) -> FailureDetector:
        d = json.loads(path.read_text())
        return FailureDetector(Logistic.from_json(d["grasp"]), Logistic.from_json(d["place"]), d["threshold"],
                               d.get("meta", {}))


def safe_retreat(env, ticks: int = 14) -> None:
    """Open the jaw slightly and rise vertically to transit height (through the same safety filter)."""
    ee, yaw, pitch = env.ee_state()
    target = np.array([ee[0], ee[1], TRANSIT_Z])
    for _ in range(ticks):
        cur, _, _ = env.ee_state()
        d = target - cur
        n = float(np.linalg.norm(d))
        step = cur + (d if n < 0.016 else d / n * 0.016)
        env.step(EEAction(step, yaw, pitch, 0.3))


class RecoveryHook:
    """step_hook for eval.runner.run_episode: retry a failed object step up to `max_retries` times."""

    def __init__(self, chore, perceiver, detector_path: str | None = None, detector: FailureDetector | None = None,
                 max_retries: int = 2, oracle: bool = False):
        self.chore = chore
        self.oracle = oracle  # upper bound: simulator truth instead of the learned detector (never deployable)
        self.perceiver = perceiver
        self.det = detector or FailureDetector.load(Path(detector_path))  # type: ignore[arg-type]
        self.max_retries = max_retries
        self.log: list[dict] = []

    def __call__(self, env, rec) -> str:
        x = step_features(rec, env.obj_by_name[rec.obj])
        failed, phase, p = self.det.judge(x)
        if self.oracle:
            failed, phase = (not rec.placed), ("grasp" if not rec.grasped else "place")
        entry = {"obj": rec.obj, "attempt": rec.attempt, "flagged": bool(failed), "phase": phase, "p_ok": p,
                 "true_placed": bool(rec.placed)}
        if failed and rec.attempt < self.max_retries:
            safe_retreat(env)
            entry["action"] = "retry"
            self.log.append(entry)
            return "retry"
        entry["action"] = "help" if failed else "next"
        self.log.append(entry)
        return "help" if failed else "next"
