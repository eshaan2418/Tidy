"""Real SO-100/SO-101 follower driver (LeRobot backend), guarded by the SafeController token.

ONLY `scripts/supervised_session.py` may construct `SO100Driver` (enforced by tests/test_safety.py).

The LeRobot API changes between releases. This adapter targets the `lerobot.robots.so100_follower`
interface (observation keys `"<joint>.pos"`, `send_action` with the same keys). It MUST be checked
against the installed version at bring-up (HUMAN_TODO H3); it has not been run against hardware.
Joint units are mapped radians <-> LeRobot degrees through `hardware/calibration.yaml`
(per-joint sign + offset, measured at bring-up). The driver refuses to start without that file.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from safety.core import GuardedDriver

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
CALIB_PATH = Path(__file__).resolve().parent / "calibration.yaml"


class CalibrationMissing(RuntimeError):
    pass


def load_calibration(path: Path = CALIB_PATH) -> dict:
    if not path.exists():
        raise CalibrationMissing(f"{path} missing: run the M1 calibration checklist (hardware/CHECKLIST.md)")
    cal = yaml.safe_load(path.read_text())
    for j in JOINTS + ["gripper"]:
        if j not in cal.get("joints", {}):
            raise CalibrationMissing(f"calibration for joint {j} missing")
    return cal


class SO100Driver(GuardedDriver):
    def __init__(self, port: str, calibration: dict | None = None, effort_frac: float = 0.5):
        super().__init__()
        self.cal = calibration or load_calibration()
        self.port = port
        self.effort_frac = effort_frac
        self._robot: Any = None
        self._last_q: np.ndarray | None = None
        self._last_g = 0.0

    # -- unit conversion -------------------------------------------------------------------------
    def _to_rad(self, j: str, deg: float) -> float:
        c = self.cal["joints"][j]
        return float(np.deg2rad(c["sign"] * (deg - c["offset_deg"])))

    def _to_deg(self, j: str, rad: float) -> float:
        c = self.cal["joints"][j]
        return float(c["sign"] * np.rad2deg(rad) + c["offset_deg"])

    def connect(self) -> None:
        from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig  # lazy

        cfg = SO100FollowerConfig(port=self.port, id=self.cal.get("robot_id", "tidy_follower"))
        self._robot = SO100Follower(cfg)
        self._robot.connect()
        # conservative current/torque limit: servo-level setting (verify register name at bring-up)
        try:
            for j in JOINTS + ["gripper"]:
                self._robot.bus.write("Torque_Limit", j, int(1000 * self.effort_frac))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"could not set torque limit; refusing to run: {e}") from e
        self.read_state()

    def read_state(self) -> tuple[np.ndarray, float]:
        assert self._robot is not None, "not connected"
        obs = self._robot.get_observation()
        q = np.array([self._to_rad(j, obs[f"{j}.pos"]) for j in JOINTS])
        gc = self.cal["joints"]["gripper"]
        g = float(np.clip((obs["gripper.pos"] - gc["open"]) / (gc["closed"] - gc["open"]), 0, 1))
        self._last_q, self._last_g = q, g
        return q, g

    def send(self, token: object, q: np.ndarray, gripper: float) -> None:
        self._check(token)
        assert self._robot is not None
        gc = self.cal["joints"]["gripper"]
        act = {f"{j}.pos": self._to_deg(j, float(v)) for j, v in zip(JOINTS, q)}
        act["gripper.pos"] = float(gc["open"] + gripper * (gc["closed"] - gc["open"]))
        self._robot.send_action(act)

    def freeze(self, token: object) -> None:
        self._check(token)
        if self._robot is None:
            return
        q, g = self.read_state()
        gc = self.cal["joints"]["gripper"]
        act = {f"{j}.pos": self._to_deg(j, float(v)) for j, v in zip(JOINTS, q)}
        act["gripper.pos"] = float(gc["open"] + g * (gc["closed"] - gc["open"]))
        self._robot.send_action(act)

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None


class Camera:
    """OpenCV USB camera with timestamped frames (latency measured in hardware/latency.py)."""

    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30):
        import cv2

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} not available")

    def read(self) -> tuple[np.ndarray | None, float]:
        import cv2

        ok, frame = self.cap.read()
        t = time.monotonic()
        if not ok:
            return None, t
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t

    def close(self) -> None:
        self.cap.release()
