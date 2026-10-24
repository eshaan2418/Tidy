"""Small geometry helpers (angles, rotations, symmetric yaw features)."""

from __future__ import annotations

import numpy as np


def wrap_pi(a: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def wrap_half_pi(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle with pi symmetry (parallel jaw / box) into [-pi/2, pi/2)."""
    return (np.asarray(a) + np.pi / 2) % np.pi - np.pi / 2


def yaw_feat(a: float | np.ndarray) -> np.ndarray:
    """pi-symmetric angle -> (cos 2a, sin 2a)."""
    a = np.asarray(a)
    return np.stack([np.cos(2 * a), np.sin(2 * a)], axis=-1)


def yaw_from_feat(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f)
    return 0.5 * np.arctan2(f[..., 1], f[..., 0])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q: np.ndarray) -> float:
    R = quat_to_mat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def min_jerk(t: np.ndarray) -> np.ndarray:
    """Minimum-jerk time scaling s(t) for t in [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return 10 * t**3 - 15 * t**4 + 6 * t**5
