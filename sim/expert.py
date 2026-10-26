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
TRANSIT_Z = 0.12  # fingertips clear the tallest object (cup, 7 cm) even on the 2.5 cm shelf


def polar_step(ee: np.ndarray, goal: np.ndarray, max_step: float, transit_z: float) -> np.ndarray:
    """One bounded step toward `goal`, moving in (radius, pan angle, z) so the path arcs around the robot
    base, and rising to `transit_z` before any long horizontal move (avoids sweeping through objects)."""
    r0, t0 = float(np.hypot(ee[0], ee[1])), float(np.arctan2(ee[1], ee[0]))
    r1, t1 = float(np.hypot(goal[0], goal[1])), float(np.arctan2(goal[1], goal[0]))
    dth = float((t1 - t0 + np.pi) % (2 * np.pi) - np.pi)
    horiz = float(np.hypot(r1 - r0, 0.5 * (r0 + r1) * dth))
    z_goal = float(goal[2])
    if horiz > 0.03 and ee[2] < transit_z - 0.005:
        z_goal = max(z_goal, transit_z)
        # rise vertically first (fingers may still straddle an item just placed), then allow a little
        # horizontal motion once the fingertips are near transit height
        dz = np.clip(z_goal - ee[2], -max_step, max_step)
        frac = 0.0 if ee[2] < transit_z - 0.012 else 0.25 * max_step / max(horiz, 1e-6)
        r, th = r0 + frac * (r1 - r0), t0 + frac * dth
        return np.array([r * np.cos(th), r * np.sin(th), ee[2] + dz])
    d = np.array([r1 - r0, 0.5 * (r0 + r1) * dth, z_goal - ee[2]])
    n = float(np.linalg.norm(d))
    s = 1.0 if n <= max_step else max_step / n
    r, th, z = r0 + s * (r1 - r0), t0 + s * dth, ee[2] + s * (z_goal - ee[2])
    return np.array([r * np.cos(th), r * np.sin(th), z])
PHASES = ("reach", "descend", "grasp", "lift", "transport", "lower", "release", "retreat")


def grasp_yaw(obj: ObjectSpec, obj_yaw: float) -> float:
    """Jaw axis must close across the object's narrow dimension."""
    if obj.kind == "box":
        return float(wrap_half_pi(obj_yaw + (np.pi / 2 if obj.size[0] > obj.size[1] else 0.0)))
    if obj.kind == "capsule":
        return float(wrap_half_pi(obj_yaw + np.pi / 2))
    return float(wrap_half_pi(obj_yaw))  # cylinder: any


def jaw_half_extent(obj: ObjectSpec) -> float:
    if obj.kind == "box":
        return float(min(obj.size[0], obj.size[1]))
    return float(obj.size[0])


def gripper_cmd_for_opening(opening: float) -> float:
    from common.kinematics import GEOM

    return float(np.clip(1.0 - opening / GEOM.max_opening, 0.0, 1.0))


def grasp_height(obj: ObjectSpec) -> float:
    return float(max(obj.height / 2, 0.014))


@dataclass
class PickPlaceTask:
    obj: ObjectSpec
    place_xyz: np.ndarray       # where the object's base centre should end up (z = support surface)
    release_clearance: float = 0.01


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
        dist = float(np.linalg.norm(goal - ee_pos))
        target = polar_step(ee_pos, goal, MAX_STEP, TRANSIT_Z)
        w = 2 * jaw_half_extent(self.t.obj)
        if self.phase in ("grasp", "lift", "transport", "lower"):
            grip = 1.0
        elif self.phase in ("release", "retreat"):
            grip = gripper_cmd_for_opening(w + 0.02)     # open just enough: don't sweep neighbours
        else:
            grip = gripper_cmd_for_opening(w + 0.035)    # pre-grasp opening
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
