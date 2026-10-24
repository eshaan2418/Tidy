"""Analytic kinematics for the SO-100-class 5-DoF arm (+ parallel gripper) used in sim and retargeting.

Conventions (robot/table frame): origin at the robot base on the table surface, x forward
(towards the human), y left, z up.  Joints: [shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll].  Lift/elbow/wrist-flex rotate about the pan-frame y axis; angles are
measured from vertical (all-zero = arm pointing straight up).  Tool pitch p: 0 = gripper
pointing straight down, +pi/2 = pointing horizontally outward along the pan direction.
Jaw yaw psi: world-frame angle of the jaw (finger-closing) axis for a top-down grasp.

Link lengths approximate the SO-100 (upper arm ~0.113 m, forearm ~0.135 m); they are NOT
the measured values of a specific unit -- real values come from calibration (hardware/).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import wrap_half_pi, wrap_pi


@dataclass(frozen=True)
class ArmGeometry:
    base_height: float = 0.05      # pan joint above table
    shoulder_height: float = 0.10  # lift joint above table
    l1: float = 0.113              # upper arm
    l2: float = 0.135              # forearm
    l3: float = 0.100              # wrist-flex pivot -> TCP (grasp centre between fingertips)
    max_opening: float = 0.08      # jaw opening, m
    # joint limits (rad), conservative hardware-ish limits; the safety layer adds margins on top
    q_min: tuple[float, ...] = (-1.92, -1.75, -0.20, -1.92, -1.66)
    q_max: tuple[float, ...] = (1.92, 1.75, 2.80, 1.92, 1.66)


GEOM = ArmGeometry()
N_ARM_JOINTS = 5


def fk(q: np.ndarray, g: ArmGeometry = GEOM) -> tuple[np.ndarray, float, float]:
    """Forward kinematics -> (tcp position xyz, jaw yaw psi (pi-symmetric), pitch p)."""
    q0, q1, q2, q3, q4 = [float(v) for v in q[:5]]
    phi1, phi2, phi3 = q1, q1 + q2, q1 + q2 + q3
    r = g.l1 * np.sin(phi1) + g.l2 * np.sin(phi2) + g.l3 * np.sin(phi3)
    z = g.shoulder_height + g.l1 * np.cos(phi1) + g.l2 * np.cos(phi2) + g.l3 * np.cos(phi3)
    pos = np.array([r * np.cos(q0), r * np.sin(q0), z])
    pitch = float(np.pi - phi3)
    psi = float(wrap_half_pi(q0 + np.pi / 2 - q4))
    return pos, psi, pitch


def wrist_point(q: np.ndarray, g: ArmGeometry = GEOM) -> np.ndarray:
    q0, q1, q2 = float(q[0]), float(q[1]), float(q[2])
    r = g.l1 * np.sin(q1) + g.l2 * np.sin(q1 + q2)
    z = g.shoulder_height + g.l1 * np.cos(q1) + g.l2 * np.cos(q1 + q2)
    return np.array([r * np.cos(q0), r * np.sin(q0), z])


def elbow_point(q: np.ndarray, g: ArmGeometry = GEOM) -> np.ndarray:
    q0, q1 = float(q[0]), float(q[1])
    r = g.l1 * np.sin(q1)
    z = g.shoulder_height + g.l1 * np.cos(q1)
    return np.array([r * np.cos(q0), r * np.sin(q0), z])


def _ik_exact(pos: np.ndarray, psi: float, pitch: float, g: ArmGeometry) -> np.ndarray | None:
    x, y, z = [float(v) for v in pos]
    r = float(np.hypot(x, y))
    if r < 1e-4:
        return None
    q0 = float(np.arctan2(y, x))
    phi3 = np.pi - pitch
    wr = r - g.l3 * np.sin(phi3)
    wz = z - g.l3 * np.cos(phi3)
    dr, dz = wr, wz - g.shoulder_height
    d2 = dr * dr + dz * dz
    c2 = (d2 - g.l1**2 - g.l2**2) / (2 * g.l1 * g.l2)
    if c2 > 1.0 or c2 < -1.0:
        return None
    q2 = float(np.arccos(c2))  # elbow-up (forward bend)
    beta = np.arctan2(dr, dz)
    q1 = float(beta - np.arctan2(g.l2 * np.sin(q2), g.l1 + g.l2 * np.cos(q2)))
    q3 = float(phi3 - q1 - q2)
    q4 = float(wrap_half_pi(q0 + np.pi / 2 - psi))
    q = np.array([q0, q1, q2, float(wrap_pi(q3)), q4])
    if np.any(q < np.array(g.q_min)) or np.any(q > np.array(g.q_max)):
        return None
    return q


@dataclass
class IKResult:
    q: np.ndarray
    feasible: bool        # exact solution at requested pose & pitch
    pitch_used: float
    pos_error: float      # distance between requested and achieved TCP position


def ik(pos: np.ndarray, psi: float, pitch: float, g: ArmGeometry = GEOM,
       pitch_search: bool = True, q_ref: np.ndarray | None = None) -> IKResult:
    """Analytic IK with feasibility fallback.

    1. try requested pitch; 2. search nearby pitches (closest first) in [0, 1.4] rad;
    3. otherwise project the target radially toward the shoulder until reachable (pos_error > 0).
    """
    q = _ik_exact(pos, psi, pitch, g)
    if q is not None:
        return IKResult(q, True, pitch, 0.0)
    if pitch_search:
        cands = np.linspace(0.0, 1.4, 29)
        for p in sorted(cands, key=lambda c: abs(c - pitch)):
            q = _ik_exact(pos, psi, float(p), g)
            if q is not None:
                return IKResult(q, False, float(p), 0.0)
    # project toward a comfortably reachable point
    target = np.asarray(pos, dtype=float).copy()
    anchor = np.array([0.15 * np.cos(np.arctan2(target[1], target[0])),
                       0.15 * np.sin(np.arctan2(target[1], target[0])), 0.08])
    for a in np.linspace(0.0, 1.0, 41)[1:]:
        cand = (1 - a) * target + a * anchor
        for p in sorted(np.linspace(0.0, 1.4, 15), key=lambda c: abs(c - pitch)):
            q = _ik_exact(cand, psi, float(p), g)
            if q is not None:
                return IKResult(q, False, float(p), float(np.linalg.norm(cand - target)))
    fallback = q_ref if q_ref is not None else HOME_Q.copy()
    return IKResult(np.asarray(fallback, dtype=float), False, pitch, float("inf"))


def preferred_pitch(pos: np.ndarray, g: ArmGeometry = GEOM) -> float:
    """Most top-down pitch that is exactly reachable at `pos` (used by retargeting/planners)."""
    for p in np.linspace(0.0, 1.4, 29):
        if _ik_exact(pos, 0.0, float(p), g) is not None:
            return float(p)
    return 1.4


HOME_Q = np.array([0.0, -0.281, 1.617, 1.806, 0.0])  # parked: TCP at (0.10, 0, 0.14), pointing down
