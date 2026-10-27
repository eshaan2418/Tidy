"""Real-robot adapter exposing the subset of sim.env.TidyEnv used by the chore executor (eval/runner.py), so the
SAME executor, policies, ready-pose primitive and recovery hook run on the real arm.

Every command goes through scripts/supervised_session.py's RealRobot (SafeController: limits, workspace box,
e-stop, watchdog, proximity pause). Object poses come from the overhead camera (RayPlanePerceiver) -- there is no
ground truth, so the executor's per-phase fields become camera-based auto-scores, confirmed by the operator.
Constructed only by eval/session_protocols.py in real mode (never by the agent).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.kinematics import GEOM, HOME_Q, fk, ik
from sim.env import EEAction, Placement
from sim.scene import FixtureSpec, ObjectSpec

CAMERA_FILE = Path(__file__).resolve().parent / "cameras" / "overhead.json"


def load_overhead_camera(path: Path = CAMERA_FILE):
    from perception.overhead import CameraModel

    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run hardware/camera_calib.py (HUMAN_TODO H2) first")
    d = json.loads(path.read_text())
    return CameraModel(np.array(d["K"]), np.array(d["R"]), np.array(d["t"]))


class RealEnv:
    def __init__(self, robot, objects: list[ObjectSpec], fixtures: list[FixtureSpec], perceiver):
        self.robot = robot
        self.objects = list(objects)
        self.all_objects = list(objects)
        self.fixtures = list(fixtures)
        self.obj_by_name = {o.name: o for o in objects}
        self.fix_by_name = {f.name: f for f in fixtures}
        self.perceiver = perceiver
        self.q_cmd = HOME_Q.copy()
        self.g_cmd = 0.0
        self.attached: str | None = None      # unknown on the real robot (no contact sensing)
        self.safety_events: list[str] = []
        self._last: dict[str, tuple[np.ndarray, float]] = {}

    # --- state
    def joint_pos(self) -> np.ndarray:
        return self.robot.joint_pos()

    def ee_state(self):
        return fk(self.joint_pos())

    def opening(self) -> float:
        _, g = self.robot.driver.read_state()
        return float((1.0 - g) * GEOM.max_opening)

    def obj_pose(self, name: str) -> tuple[np.ndarray, float]:
        seen = self.perceiver.perceive(self, [name])
        self._last.update(seen)
        return seen[name][0].copy(), float(seen[name][1])

    def obj_upright(self, name: str) -> bool:
        return True   # not observable from a top-down RGB camera; the operator confirms in scoring

    def render(self, camera: str = "overhead", width: int = 320, height: int = 240, depth: bool = False):
        import cv2

        f = self.robot.frame()
        f = np.zeros((height, width, 3), np.uint8) if f is None else cv2.resize(f, (width, height))
        return (f, None) if depth else f

    # --- motion (through the SafeController)
    def reset(self, placement: Placement, settle: bool = True) -> None:
        q0 = self.joint_pos()
        for a in np.linspace(0, 1, 30)[1:]:           # slow move home, 3 s
            self.robot.command_joints((1 - a) * q0 + a * HOME_Q, 0.0)
        self.q_cmd, self.g_cmd = HOME_Q.copy(), 0.0
        self.safety_events = []

    def step(self, act: EEAction, monitor_rgb=None) -> dict:
        res = ik(np.asarray(act.pos, float), float(act.yaw), float(act.pitch), q_ref=self.q_cmd)
        status = self.robot.command_joints(res.q, float(np.clip(act.gripper, 0, 1)))
        if status != "ok":
            self.safety_events.append(status)
        self.q_cmd, self.g_cmd = res.q.copy(), float(act.gripper)
        return {"status": status, "ik_feasible": res.feasible, "ik_pos_err": res.pos_error,
                "grasped": None, "released": None}

    def settle(self, steps: int = 100) -> None:
        import time

        time.sleep(min(2.0, steps * 0.002))

    def close(self) -> None:
        pass
