"""Closed-loop scripted pick-and-place controller in the robot's own action space.

Used for (a) the *simulated teleoperator* baseline (with operator noise models layered on top in
sim/teleop.py) and (b) sanity tests of the environment.  It is NOT a learned policy and is never
reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import wrap_half_pi
from common.kinematics import preferred_pitch
from sim.env import EEAction
from sim.scene import ObjectSpec

MAX_STEP = 0.018  # m per 10 Hz tick (just under the 0.2 m/s TCP safety limit)
PHASES = ("reach", "descend", "grasp", "lift", "transport", "lower", "release", "retreat")


def grasp_yaw(obj: ObjectSpec, obj_yaw: float) -> float:
    """Jaw axis must close across the object's narrow dimension."""
    if obj.kind == "box":
        return float(wrap_half_pi(obj_yaw + (np.pi / 2 if obj.size[0] > obj.size[1] else 0.0)))
    if obj.kind == "capsule":
        return float(wrap_half_pi(obj_yaw + np.pi / 2))
    return float(wrap_half_pi(obj_yaw))  # cylinder: any


def grasp_height(obj: ObjectSpec) -> float:
    return float(max(obj.height / 2, 0.014))


@dataclass
class PickPlaceTask:
    obj: ObjectSpec
    place_xyz: np.ndarray       # where the object's base centre should end up (z = support surface)
    release_clearance: float = 0.02


class ScriptedPickPlace:
    def __init__(self, task: PickPlaceTask, obj_pos: np.ndarray, obj_yaw: float, hover: float = 0.07):
        self.t = task
        self.hover = hover
        self.phase = "reach"
        self.ticks = 0
        self.obj_pos = np.asarray(obj_pos, dtype=float)
        self.yaw = grasp_yaw(task.obj, obj_yaw)
        self.gz = grasp_height(task.obj)
        self.done = False
        self._last_ee: np.ndarray | None = None
        self._stall = 0

    def update_object(self, obj_pos: np.ndarray, obj_yaw: float) -> None:
        if self.phase in ("reach", "descend"):
            self.obj_pos = np.asarray(obj_pos, dtype=float)
            self.yaw = grasp_yaw(self.t.obj, obj_yaw)

    def _goal(self) -> tuple[np.ndarray, float]:
        o, gz, h = self.obj_pos, self.gz, self.hover
        pl = self.t.place_xyz
        carry_z = max(pl[2] + self.t.obj.height + 0.05, 0.09)
        if self.phase == "reach":
            return np.array([o[0], o[1], gz + h]), 0.012
        if self.phase == "descend":
            return np.array([o[0], o[1], gz]), 0.004
        if self.phase in ("grasp",):
            return np.array([o[0], o[1], gz]), 1.0
        if self.phase == "lift":
            return np.array([o[0], o[1], carry_z]), 0.012
        if self.phase == "transport":
            return np.array([pl[0], pl[1], carry_z]), 0.010
        if self.phase == "lower":
            return np.array([pl[0], pl[1], pl[2] + self.gz + self.t.release_clearance]), 0.006
        if self.phase == "release":
            return np.array([pl[0], pl[1], pl[2] + self.gz + self.t.release_clearance]), 1.0
        return np.array([pl[0], pl[1], carry_z + 0.02]), 0.012

    def act(self, ee_pos: np.ndarray) -> EEAction:
        goal, tol = self._goal()
        delta = goal - ee_pos
        dist = float(np.linalg.norm(delta))
        step = delta if dist <= MAX_STEP else delta / dist * MAX_STEP
        target = ee_pos + step
        grip = 1.0 if self.phase in ("grasp", "lift", "transport", "lower") else 0.0
        self.ticks += 1
        if self._last_ee is not None and np.linalg.norm(ee_pos - self._last_ee) < 0.001:
            self._stall += 1
        else:
            self._stall = 0
        self._last_ee = np.asarray(ee_pos, dtype=float).copy()
        advance = False
        if self.phase in ("grasp", "release"):
            advance = self.ticks >= 7  # gripper is rate-limited by the safety filter
        elif dist < tol:
            advance = True
        elif self._stall >= 3 and self.ticks > 3:
            advance = True  # blocked (e.g., fingertips touching the table) -> proceed
        if self.ticks > 80:
            advance = True  # timeout -> move on (failure will be scored)
        if advance:
            i = PHASES.index(self.phase)
            if i + 1 < len(PHASES):
                self.phase = PHASES[i + 1]
                self.ticks = 0
                self._stall = 0
            else:
                self.done = True
        pitch = preferred_pitch(target)
        return EEAction(target, self.yaw, pitch, grip)
