"""Interactive joint calibration (torque OFF) -> hardware/calibration.yaml. Human-operated."""

from __future__ import annotations

import argparse

import yaml

from hardware.so100_driver import CALIB_PATH, JOINTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    a = ap.parse_args()
    from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

    r = SO100Follower(SO100FollowerConfig(port=a.port, id="tidy_follower"))
    r.connect(calibrate=False)
    r.bus.disable_torque()

    def read() -> dict:
        return r.get_observation()

    input("Move the arm to the ZERO pose (straight up, jaw axis left-right). Press Enter.")
    z = read()
    cal: dict = {"robot_id": "tidy_follower", "joints": {}}
    for j in JOINTS:
        input(f"Move {j} a few degrees in its POSITIVE direction (see CHECKLIST C.2). Press Enter.")
        d = read()[f"{j}.pos"] - z[f"{j}.pos"]
        cal["joints"][j] = {"offset_deg": float(z[f"{j}.pos"]), "sign": 1 if d > 0 else -1}
        input(f"Return {j} to zero. Press Enter.")
    input("Open the gripper fully. Press Enter.")
    g_open = read()["gripper.pos"]
    input("Close the gripper fully. Press Enter.")
    g_closed = read()["gripper.pos"]
    cal["joints"]["gripper"] = {"open": float(g_open), "closed": float(g_closed), "offset_deg": 0.0, "sign": 1}
    CALIB_PATH.write_text(yaml.safe_dump(cal))
    print("wrote", CALIB_PATH)
    r.disconnect()


if __name__ == "__main__":
    main()
