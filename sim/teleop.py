"""Simulated teleoperator (leader-arm operator model) for the sim-stage teleop baseline.

The operator watches the scene directly, so it uses the true object pose with a small visual-judgement
error (not the robot's perception). Noise model, applied on top of the scripted controller (sim/expert.py):
  * Ornstein-Uhlenbeck tracking noise on the commanded TCP position (sigma ~3 mm)
  * per-object yaw misjudgement (sigma 5 deg) and position misjudgement (sigma 3 mm)
  * slower-than-max speed (leader arms are moved deliberately), varying per demo
  * occasional hover/correction pauses before descending
These parameters are ASSUMPTIONS (no real teleop data yet), and the M2/M4 results report a sweep over them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import wrap_half_pi
from policies.spec import View
from sim.env import EEAction
from sim.expert import PickPlaceTask, ScriptedPickPlace
from sim.scene import ObjectSpec


@dataclass
class OperatorModel:
    track_sigma: float = 0.003
    pos_judgement_sigma: float = 0.003
    yaw_judgement_sigma: float = np.deg2rad(5)
    speed_range: tuple[float, float] = (0.6, 0.9)   # fraction of the max TCP step
    hesitate_prob: float = 0.25
    hesitate_ticks: tuple[int, int] = (3, 10)


class TeleopAgent:
    def __init__(self, env_truth, model: OperatorModel | None = None, seed: int = 0):
        self.truth = env_truth          # operator looks at the real scene
        self.m = model or OperatorModel()
        self.rng = np.random.default_rng(seed)
        self.ctl: ScriptedPickPlace | None = None
        self.ou = np.zeros(3)
        self.hesitate = 0
        self.speed = 1.0

    def begin_step(self, obj: ObjectSpec, obj_pos: np.ndarray, obj_yaw: float, goal: np.ndarray) -> None:
        p, y = self.truth.obj_pose(obj.name)
        p = p + np.r_[self.rng.normal(0, self.m.pos_judgement_sigma, 2), 0.0]
        y = float(wrap_half_pi(y + self.rng.normal(0, self.m.yaw_judgement_sigma)))
        self.ctl = ScriptedPickPlace(PickPlaceTask(obj, np.asarray(goal, dtype=float)), p, y)
        self.speed = float(self.rng.uniform(*self.m.speed_range))
        hesitates = self.rng.random() < self.m.hesitate_prob
        self.hesitate = int(self.rng.integers(*self.m.hesitate_ticks)) if hesitates else 0
        self.ou = np.zeros(3)

    def act(self, view: View) -> EEAction:
        assert self.ctl is not None
        ee = view.ee_pos
        a = self.ctl.act(ee)
        if self.ctl.phase == "descend" and self.hesitate > 0:
            self.hesitate -= 1
            a = EEAction(ee + self.rng.normal(0, 0.001, 3), a.yaw, a.pitch, a.gripper)
        d = (a.pos - ee) * self.speed
        # stationary OU tracking error around the intended path: command the *increment* of the offset so it
        # does not integrate into drift
        prev = self.ou.copy()
        self.ou += -0.3 * self.ou + self.rng.normal(0, self.m.track_sigma * 0.5, 3)
        return EEAction(ee + d + (self.ou - prev) * np.array([1, 1, 0.5]), a.yaw, a.pitch, a.gripper)

    @property
    def done(self) -> bool:
        return bool(self.ctl and self.ctl.done)
