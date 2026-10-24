"""Conservative safety limits shared by sim and the real robot.

All numbers here are deliberately conservative defaults for an SO-100-class arm on a table.
They must be re-checked against the specific unit at hardware bring-up (HUMAN_TODO M1).
Changing them requires a DECISIONS.md entry; tests in tests/test_safety.py pin the bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.kinematics import GEOM


@dataclass(frozen=True)
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        p = np.asarray(p)
        return bool(np.all(p >= np.array(self.lo) + margin) and np.all(p <= np.array(self.hi) - margin))

    def clamp(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), self.lo, self.hi)


@dataclass(frozen=True)
class SafetyConfig:
    # joint position limits = hardware limits shrunk by a margin (rad)
    joint_margin: float = 0.10
    # max joint speed (rad/s) per joint [pan, lift, elbow, wrist_flex, roll]
    max_joint_vel: tuple[float, ...] = (1.2, 1.0, 1.2, 1.8, 2.5)
    max_gripper_vel: float = 0.10                 # m/s of jaw opening
    max_tcp_speed: float = 0.20                   # m/s Cartesian TCP speed
    # max motor effort as a fraction of rated torque; real driver maps to current limit
    max_effort_frac: float = 0.5
    sim_force_limit: float = 6.0                  # N*m, matches actuator forcerange in sim/scene.py
    # Cartesian workspace box for the TCP (table frame, metres). z floor keeps fingertips (TCP-0.012 m) off the table.
    workspace: Box = Box((-0.02, -0.30, 0.013), (0.31, 0.30, 0.22))
    # no-go zones the TCP may never enter (e.g., beyond the table edge on the human side, robot base column)
    no_go: tuple[Box, ...] = field(default_factory=lambda: (
        Box((0.31, -1.0, -1.0), (2.0, 1.0, 2.0)),          # beyond far table edge (human side)
        Box((-0.06, -0.06, -1.0), (0.06, 0.06, 0.20)),     # robot base column
    ))
    # watchdog: policy/perception heartbeat must arrive within this many seconds
    watchdog_timeout_s: float = 0.5
    # proximity: pause if a human hand is detected inside the workspace while the robot is moving
    proximity_min_pixels: int = 40
    proximity_resume_clear_s: float = 2.0
    # operator presence must be re-confirmed this often during supervised sessions
    presence_period_s: float = 300.0

    @property
    def q_lo(self) -> np.ndarray:
        return np.array(GEOM.q_min) + self.joint_margin

    @property
    def q_hi(self) -> np.ndarray:
        return np.array(GEOM.q_max) - self.joint_margin


DEFAULT_SAFETY = SafetyConfig()


def in_no_go(p: np.ndarray, cfg: SafetyConfig = DEFAULT_SAFETY) -> bool:
    return any(z.contains(p) for z in cfg.no_go)
