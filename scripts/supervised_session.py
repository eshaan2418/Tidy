"""THE ONLY entry point that may move the real robot.

    python scripts/supervised_session.py safety-smoke            # M1: scripted slow motions + e-stop drills
    python scripts/supervised_session.py eval --session S2       # queued evaluation session (SESSION_QUEUE.md)
    python scripts/supervised_session.py practice --session P1   # supervised self-practice
    add --sim to run the identical protocol against the MuJoCo twin (used by CI / the agent)

Real mode refuses to run unless:
  * stdin is an interactive TTY and the operator types the presence phrase,
  * SAFETY_LOG.md has no unresolved entries,
  * hardware/calibration.yaml exists,
and it re-confirms operator presence before every trial and at least every 5 minutes.
Every trial is recorded to video under data/private/sessions/<timestamp>/.
An automated agent must never run real mode (it cannot pass the interactive checks by design).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.kinematics import HOME_Q, ik  # noqa: E402
from safety.core import (  # noqa: E402
    FileEStop,
    KeyboardEStop,
    LatchingEStop,
    ProximityMonitor,
    SafeController,
    Watchdog,
    estop_file_default,
)
from safety.limits import DEFAULT_SAFETY  # noqa: E402

PRESENCE_PHRASE = "i am present and watching the robot"
SAFETY_LOG = ROOT / "SAFETY_LOG.md"
SESSIONS_DIR = ROOT / "data" / "private" / "sessions"


class SessionAbort(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------- gating
def require_safety_log_resolved(path: Path = SAFETY_LOG) -> None:
    """Every entry line `- [ ] ...` (open) blocks the next session; resolved entries are `- [x] ... resolved:`."""
    text = path.read_text() if path.exists() else ""
    open_items = [ln for ln in text.splitlines() if re.match(r"^\s*- \[ \]", ln)]
    if open_items:
        raise SessionAbort(f"SAFETY_LOG.md has {len(open_items)} unresolved entries; review and resolve first")


def confirm_operator_presence(interactive: bool, prompt: str = "") -> None:
    if not interactive:
        return  # sim mode
    if not sys.stdin.isatty():
        raise SessionAbort("real-robot sessions require an interactive operator terminal")
    print(prompt or "Operator: confirm you are at the robot, the area is clear, and the e-stop is in reach.")
    ans = input(f'Type exactly "{PRESENCE_PHRASE}": ').strip().lower()
    if ans != PRESENCE_PHRASE:
        raise SessionAbort("operator presence not confirmed")


def append_safety_log(kind: str, detail: str, session: str) -> None:
    ts = dt.datetime.now().isoformat(timespec="seconds")
    with SAFETY_LOG.open("a") as f:
        f.write(f"- [ ] {ts} session={session} {kind}: {detail}\n")


# ----------------------------------------------------------------------------------------- robots
class RealRobot:
    """Real arm + cameras wrapped in the SafeController; mirrors the subset of TidyEnv used by sessions."""

    def __init__(self, port: str, monitor_cam: int, scene_cam: int, session: str):
        from hardware.so100_driver import Camera, SO100Driver, load_calibration

        self.session = session
        cal = load_calibration()
        self.driver = SO100Driver(port, cal, effort_frac=DEFAULT_SAFETY.max_effort_frac)
        self.driver.connect()
        self.monitor = Camera(monitor_cam)
        self.scene = Camera(scene_cam) if scene_cam != monitor_cam else self.monitor
        es = LatchingEStop([KeyboardEStop(), FileEStop(estop_file_default())])
        self.ctrl = SafeController(self.driver, es, Watchdog(DEFAULT_SAFETY.watchdog_timeout_s),
                                   ProximityMonitor(DEFAULT_SAFETY), DEFAULT_SAFETY,
                                   log=lambda s: append_safety_log("event", s, session))

    def joint_pos(self) -> np.ndarray:
        return self.driver.read_state()[0]

    def command_joints(self, q: np.ndarray, g: float, dt_: float = 0.1) -> str:
        rgb, _ = self.monitor.read()
        self.ctrl.watchdog.heartbeat()
        st = self.ctrl.tick(q, g, dt_, monitor_rgb=rgb)
        time.sleep(dt_)
        return st.reason

    def frame(self) -> np.ndarray | None:
        return self.scene.read()[0]

    def close(self) -> None:
        self.driver.disconnect()


class SimRobot:
    def __init__(self, session: str):
        from sim.env import Placement, TidyEnv
        from sim.scene import FIXTURES, LIB

        self.env = TidyEnv([LIB["cup"]], [FIXTURES["bin"]])
        self.env.reset(Placement({"cup": (0.18, 0.0, 0.0)}))
        self.ctrl = self.env.safety
        self.session = session

    def joint_pos(self) -> np.ndarray:
        return self.env.joint_pos()

    def command_joints(self, q: np.ndarray, g: float, dt_: float = 0.1) -> str:
        self.env.watchdog.heartbeat()
        st = self.ctrl.tick(q, g, dt_)
        self.env.settle(50)
        return st.reason

    def frame(self) -> np.ndarray | None:
        return self.env.render("overhead", 320, 240)

    def close(self) -> None:
        self.env.close()


# ----------------------------------------------------------------------------------------- recording
class TrialRecorder:
    def __init__(self, out_dir: Path, name: str, fps: int = 10):
        import imageio.v2 as imageio

        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{name}.mp4"
        self.w = imageio.get_writer(self.path, fps=fps, macro_block_size=1)

    def add(self, frame: np.ndarray | None) -> None:
        if frame is not None:
            self.w.append_data(frame)

    def close(self) -> str:
        self.w.close()
        return str(self.path)


# ----------------------------------------------------------------------------------------- protocols
SMOKE_WAYPOINTS = [  # slow, high, centre-of-workspace motions only
    (np.array([0.12, 0.00, 0.14]), 0.0), (np.array([0.18, 0.06, 0.12]), 0.3),
    (np.array([0.18, -0.06, 0.12]), -0.3), (np.array([0.14, 0.0, 0.10]), 0.0),
]


def run_safety_smoke(robot, out_dir: Path, interactive: bool) -> dict:
    log: dict = {"waypoints": [], "estop_drills": [], "limit_probes": []}
    rec = TrialRecorder(out_dir, "safety_smoke")
    try:
        confirm_operator_presence(interactive, "Safety smoke: the arm will move slowly between 4 high poses.")
        for p, yaw in SMOKE_WAYPOINTS:
            res = ik(p, yaw, 0.0)
            for _ in range(40):
                st = robot.command_joints(res.q, 0.0)
                rec.add(robot.frame())
                if st == "estop":
                    break
            log["waypoints"].append({"target": p.tolist(), "reached_q": robot.joint_pos().tolist(), "status": st})
        # e-stop drills: operator (or sim) presses e-stop while moving; arm must freeze within one tick
        for k in range(5 if interactive else 2):
            if interactive:
                confirm_operator_presence(True, f"E-stop drill {k+1}/5: press the e-stop when the arm starts moving.")
            target = ik(SMOKE_WAYPOINTS[k % 2][0], 0.0, 0.0).q
            t0, stopped_at = time.monotonic(), None
            for i in range(60):
                if not interactive and i == 5:
                    robot.env.estop_button.pressed = True
                st = robot.command_joints(target, 0.0)
                rec.add(robot.frame())
                if st == "estop":
                    stopped_at = time.monotonic() - t0
                    break
            q_a = robot.joint_pos()
            for _ in range(5):
                robot.command_joints(target, 0.0)
            drift = float(np.max(np.abs(robot.joint_pos() - q_a)))
            log["estop_drills"].append({"stopped": stopped_at is not None, "t_s": stopped_at, "drift_rad": drift})
            if interactive:
                ans = input(f'Release the e-stop, then type "{LatchingEStop.RESET_PHRASE}": ')
                if not robot.ctrl.estop.reset(ans):
                    raise SessionAbort("e-stop reset not confirmed")
            else:
                robot.env.estop_button.pressed = False
                robot.ctrl.estop.reset(LatchingEStop.RESET_PHRASE)
        # limit probes: command poses outside the workspace; filter must hold
        from common.kinematics import fk

        dive = HOME_Q.copy()
        dive[1] = 1.7  # raw joint command that would drive the arm into the table
        probes = [("tcp", np.array([0.40, 0.0, 0.05])), ("tcp", np.array([0.15, 0.0, -0.05])),
                  ("joint", dive), ("joint", np.array([3.0, 0.0, 0.0, 0.0, 0.0]))]
        for kind, bad in probes:
            q_bad = ik(bad, 0.0, 0.0).q if kind == "tcp" else bad
            statuses = [robot.command_joints(q_bad, 0.0) for _ in range(15)]
            p = fk(robot.joint_pos())[0]
            log["limit_probes"].append({"kind": kind, "requested": bad.tolist(), "statuses": sorted(set(statuses)),
                                        "tcp_after": p.tolist(),
                                        "inside_workspace": DEFAULT_SAFETY.workspace.contains(p)})
            for _ in range(15):
                robot.command_joints(HOME_Q, 0.0)
        for _ in range(30):
            robot.command_joints(HOME_Q, 0.0)
    finally:
        log["video"] = rec.close()
    log["pass"] = (all(d["stopped"] and d["drift_rad"] < 0.02 for d in log["estop_drills"])
                   and all(p["inside_workspace"] for p in log["limit_probes"]))
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol", choices=["safety-smoke", "eval", "practice"])
    ap.add_argument("--sim", action="store_true", help="run the identical protocol in the MuJoCo twin")
    ap.add_argument("--session", default="")
    ap.add_argument("--port", default="/dev/tty.usbmodem0")
    ap.add_argument("--monitor-cam", type=int, default=0)
    ap.add_argument("--scene-cam", type=int, default=0)
    a = ap.parse_args(argv)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session = a.session or f"{a.protocol}_{stamp}"
    out_dir = (ROOT / "results" / "sim_sessions" / session) if a.sim else (SESSIONS_DIR / session)
    interactive = not a.sim
    robot: RealRobot | SimRobot
    if interactive:
        require_safety_log_resolved()
        confirm_operator_presence(True)
        robot = RealRobot(a.port, a.monitor_cam, a.scene_cam, session)
    else:
        robot = SimRobot(session)
    try:
        if a.protocol == "safety-smoke":
            log = run_safety_smoke(robot, out_dir, interactive)
        else:
            from eval.session_protocols import run_queued_session

            log = run_queued_session(a.protocol, session, robot, out_dir, interactive,
                                     confirm=confirm_operator_presence, recorder_cls=TrialRecorder)
    except SessionAbort as e:
        print(f"ABORTED: {e}")
        if interactive:
            append_safety_log("abort", str(e), session)
        return 2
    finally:
        robot.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "trials"}, indent=1, default=str)[:2000])
    return 0 if log.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
