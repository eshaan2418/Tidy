"""Read-only driver check (torque stays OFF): prints raw LeRobot observation keys/values for 3 s."""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    a = ap.parse_args()
    import lerobot
    from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

    print("lerobot", getattr(lerobot, "__version__", "?"))
    r = SO100Follower(SO100FollowerConfig(port=a.port, id="tidy_follower"))
    r.connect(calibrate=False)
    try:
        r.bus.disable_torque()
        t0 = time.time()
        while time.time() - t0 < 3:
            print(r.get_observation())
            time.sleep(0.5)
    finally:
        r.disconnect()


if __name__ == "__main__":
    main()
