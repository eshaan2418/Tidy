"""Observation / action spec shared by every data source (teleop, video pseudo-demos, practice) and runtime.

Observation (object-centric, D-005), all positions relative to the TCP so policies generalise across
placements:
  ee_pos(3) | ee_yaw_feat(2) | pitch(1) | opening(1) | grip_cmd(1) |
  obj_rel(3) | obj_yaw_feat(2) | category one-hot(7) | size feats(3) | goal_rel(3)       -> 26
Action: delta TCP position / 0.02 (3) | jaw yaw feat (2) | pitch (1) | gripper cmd (1)  -> 7
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import yaw_feat, yaw_from_feat
from common.kinematics import GEOM
from sim.env import EEAction
from sim.scene import ObjectSpec

CATEGORIES = ("pen", "cup", "tape", "box", "block", "plate", "can")
OBS_DIM = 26
ACT_DIM = 7
DELTA_SCALE = 0.02


@dataclass
class View:
    """What the policy sees at one control tick (from sim GT+noise, sim perception, video, or the real robot)."""
    ee_pos: np.ndarray
    ee_yaw: float
    pitch: float
    opening: float
    grip_cmd: float
    obj_pos: np.ndarray       # perceived target object centre
    obj_yaw: float
    obj: ObjectSpec           # category / size known from perception (class) + library
    goal_pos: np.ndarray      # perceived place target (object base position)


def size_feats(o: ObjectSpec) -> np.ndarray:
    if o.kind == "box":
        long, short = max(o.size[0], o.size[1]), min(o.size[0], o.size[1])
    elif o.kind == "cylinder":
        long = short = o.size[0]
    else:
        long, short = o.size[1] + o.size[0], o.size[0]
    return np.array([o.height, long, short]) * 10.0


def obj_yaw_feat(o: ObjectSpec, yaw: float) -> np.ndarray:
    """Upright cylinders are rotationally symmetric: their perceived yaw is noise, so it is not observed."""
    if o.kind == "cylinder":
        return np.zeros(2)
    return yaw_feat(yaw)


def obs_vector(v: View) -> np.ndarray:
    cat = np.zeros(len(CATEGORIES))
    cat[CATEGORIES.index(v.obj.category)] = 1.0
    return np.concatenate([
        v.ee_pos * 5.0, yaw_feat(v.ee_yaw), [v.pitch], [v.opening / GEOM.max_opening], [v.grip_cmd],
        (v.obj_pos - v.ee_pos) * 10.0, obj_yaw_feat(v.obj, v.obj_yaw), cat, size_feats(v.obj),
        (v.goal_pos - v.ee_pos) * 10.0,
    ]).astype(np.float32)


def action_vector(ee_pos: np.ndarray, a: EEAction) -> np.ndarray:
    d = np.clip((np.asarray(a.pos) - ee_pos) / DELTA_SCALE, -1.5, 1.5)
    return np.concatenate([d, yaw_feat(a.yaw), [a.pitch], [a.gripper]]).astype(np.float32)


def action_from_vector(ee_pos: np.ndarray, x: np.ndarray) -> EEAction:
    x = np.asarray(x, dtype=float)
    d = np.clip(x[:3], -1.0, 1.0) * DELTA_SCALE
    return EEAction(ee_pos + d, float(yaw_from_feat(x[3:5])), float(np.clip(x[5], 0.0, 1.4)),
                    float(np.clip(x[6], 0.0, 1.0)))
