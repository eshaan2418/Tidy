"""Safety layer: limit filter, latching e-stop, watchdog, proximity pause, and the SafeController.

Every command to any robot (sim or real) goes through `SafeController.tick`, which, in order:
  1. checks the e-stop (latching)          -> freeze
  2. checks the watchdog heartbeat         -> freeze
  3. checks the person-proximity monitor   -> pause (hold position)
  4. filters the command (joint limits, joint/TCP rate limits, workspace box, no-go zones,
     whole-arm table clearance) and only then forwards it to the driver.
Drivers refuse commands that do not carry the controller's private token, so the filter
cannot be bypassed by calling the driver directly (tested in tests/test_safety.py).
"""

from __future__ import annotations

import os
import select
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from common.kinematics import elbow_point, fk, wrist_point
from safety.limits import DEFAULT_SAFETY, SafetyConfig, in_no_go

Clock = Callable[[], float]


# --------------------------------------------------------------------------------------------- filter
@dataclass
class FilterEvent:
    kind: str
    detail: str


@dataclass
class FilteredCommand:
    q: np.ndarray
    gripper: float
    events: list[FilterEvent] = field(default_factory=list)
    held: bool = False  # True if the command was rejected and the current pose is held


class SafetyFilter:
    def __init__(self, cfg: SafetyConfig = DEFAULT_SAFETY):
        self.cfg = cfg

    def pose_ok(self, q: np.ndarray) -> tuple[bool, str]:
        p, _, _ = fk(q)
        if not self.cfg.workspace.contains(p):
            return False, f"tcp {np.round(p, 3).tolist()} outside workspace"
        if in_no_go(p, self.cfg):
            return False, f"tcp {np.round(p, 3).tolist()} in no-go zone"
        for name, pt in (("elbow", elbow_point(q)), ("wrist", wrist_point(q))):
            if pt[2] < 0.02:
                return False, f"{name} below table clearance"
        return True, ""

    def filter(self, q_target: np.ndarray, gripper_target: float, q_current: np.ndarray,
               gripper_current: float, dt: float) -> FilteredCommand:
        cfg = self.cfg
        ev: list[FilterEvent] = []
        q_cur = np.asarray(q_current, dtype=float)[:5]
        q_t = np.asarray(q_target, dtype=float)[:5]
        if not np.all(np.isfinite(q_t)) or not np.isfinite(gripper_target):
            nan_ev = [FilterEvent("nan", "non-finite command")]
            return FilteredCommand(q_cur.copy(), float(gripper_current), nan_ev, True)
        # 1. joint position limits
        q_c = np.clip(q_t, cfg.q_lo, cfg.q_hi)
        if np.any(q_c != q_t):
            ev.append(FilterEvent("joint_limit", "target clipped to joint limits"))
        # 2. joint rate limit
        dq_max = np.array(cfg.max_joint_vel) * dt
        dq = np.clip(q_c - q_cur, -dq_max, dq_max)
        if np.any(np.abs(q_c - q_cur) > dq_max + 1e-12):
            ev.append(FilterEvent("joint_rate", "joint step rate-limited"))
        q_new = q_cur + dq
        # 3. TCP speed limit (scale joint step down until TCP displacement within bound)
        p_cur, _, _ = fk(q_cur)
        p_new, _, _ = fk(q_new)
        step = float(np.linalg.norm(p_new - p_cur))
        max_step = cfg.max_tcp_speed * dt
        if step > max_step:
            s = 1.0
            for _ in range(20):
                s *= 0.7
                q_try = q_cur + s * dq
                if np.linalg.norm(fk(q_try)[0] - p_cur) <= max_step:
                    break
            q_new = q_cur + s * dq
            ev.append(FilterEvent("tcp_speed", f"tcp step {step:.4f} m scaled"))
        # 4. workspace / no-go / clearance: reject (hold) rather than guess
        ok, why = self.pose_ok(q_new)
        held = False
        if not ok:
            cur_ok, _ = self.pose_ok(q_cur)
            ev.append(FilterEvent("workspace", why))
            if cur_ok:
                q_new = q_cur.copy()
                held = True
            else:
                # recovering from an out-of-bounds state: only accept motion that reduces violation
                q_new = q_cur.copy()
                held = True
        # 5. gripper limits + rate
        g = float(np.clip(gripper_target, 0.0, 1.0))
        gmax = cfg.max_gripper_vel * dt / 0.08
        g = float(np.clip(g, gripper_current - gmax, gripper_current + gmax))
        return FilteredCommand(q_new, g, ev, held)


# --------------------------------------------------------------------------------------------- e-stop
class EStopSource(Protocol):
    def poll(self) -> bool: ...


class FileEStop:
    """E-stop triggered by the existence of a file (used by tests and by the session GUI/CLI)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def poll(self) -> bool:
        return self.path.exists()


class KeyboardEStop:
    """Non-blocking stdin listener: any line containing 's', 'q', ' ' or ESC triggers stop.

    A USB e-stop button that enumerates as a keyboard (most do) works through this path.
    """

    def __init__(self) -> None:
        self._hit = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    line = sys.stdin.readline()
                    if any(c in line for c in ("s", "q", " ", "\x1b")) or line == "\n":
                        self._hit = True
            except Exception:
                self._hit = True  # stdin broken -> fail safe
                return

    def poll(self) -> bool:
        return self._hit

    def close(self) -> None:
        self._stop = True


class ManualEStop:
    def __init__(self) -> None:
        self.pressed = False

    def poll(self) -> bool:
        return self.pressed


class LatchingEStop:
    """Combines sources; once tripped stays tripped until `reset(confirm=...)` with the operator phrase."""

    RESET_PHRASE = "operator present, area clear"

    def __init__(self, sources: list[EStopSource]):
        if not sources:
            raise ValueError("at least one e-stop source is required")
        self.sources = sources
        self.tripped = False

    def check(self) -> bool:
        if not self.tripped and any(s.poll() for s in self.sources):
            self.tripped = True
        return self.tripped

    def reset(self, confirm: str) -> bool:
        if confirm.strip().lower() != self.RESET_PHRASE:
            return False
        if any(s.poll() for s in self.sources):
            return False
        self.tripped = False
        return True


# --------------------------------------------------------------------------------------------- watchdog
class Watchdog:
    def __init__(self, timeout_s: float, clock: Clock = time.monotonic):
        self.timeout_s = timeout_s
        self.clock = clock
        self.last = clock()

    def heartbeat(self) -> None:
        self.last = self.clock()

    def ok(self) -> bool:
        return (self.clock() - self.last) <= self.timeout_s


# --------------------------------------------------------------------------------------------- proximity
SKIN_LO = np.array([0, 40, 60])     # HSV (OpenCV scale) conservative skin range
SKIN_HI = np.array([25, 200, 255])


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, SKIN_LO, SKIN_HI) > 0


class ProximityMonitor:
    """Pause if skin-coloured pixels (a person's hand/arm) appear inside the workspace ROI while moving.

    Conservative by design: false positives just pause the robot.  `roi` is a boolean image mask of
    the workspace region in the monitoring camera.  A learned hand detector can replace `detector`.
    """

    def __init__(self, cfg: SafetyConfig = DEFAULT_SAFETY, roi: np.ndarray | None = None,
                 detector: Callable[[np.ndarray], np.ndarray] = skin_mask, clock: Clock = time.monotonic):
        self.cfg = cfg
        self.roi = roi
        self.detector = detector
        self.clock = clock
        self.paused = False
        self._clear_since: float | None = None
        self.last_count = 0

    def update(self, rgb: np.ndarray | None, robot_moving: bool) -> bool:
        if rgb is None:
            # perception failure: treat as unsafe (fail closed)
            self.paused = True
            self._clear_since = None
            return self.paused
        m = self.detector(rgb)
        if self.roi is not None:
            m = m & self.roi
        self.last_count = int(m.sum())
        present = self.last_count >= self.cfg.proximity_min_pixels
        now = self.clock()
        if present and (robot_moving or self.paused):
            self.paused = True
            self._clear_since = None
        elif self.paused:
            if self._clear_since is None:
                self._clear_since = now
            elif now - self._clear_since >= self.cfg.proximity_resume_clear_s:
                self.paused = False
        return self.paused


# --------------------------------------------------------------------------------------------- controller
class _Token:
    __slots__ = ()


class RobotDriver(Protocol):
    def read_state(self) -> tuple[np.ndarray, float]: ...
    def send(self, token: object, q: np.ndarray, gripper: float) -> None: ...
    def freeze(self, token: object) -> None: ...


class GuardedDriver:
    """Base class for drivers: rejects commands without the SafeController token."""

    def __init__(self) -> None:
        self._token: object | None = None

    def bind(self, token: object) -> None:
        if self._token is not None:
            raise RuntimeError("driver already bound to a SafeController")
        self._token = token

    def _check(self, token: object) -> None:
        if self._token is None or token is not self._token:
            raise PermissionError("robot commands must go through safety.core.SafeController")

    def read_state(self) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def send(self, token: object, q: np.ndarray, gripper: float) -> None:
        raise NotImplementedError

    def freeze(self, token: object) -> None:
        raise NotImplementedError


@dataclass
class TickStatus:
    sent: bool
    reason: str
    events: list[FilterEvent] = field(default_factory=list)


class SafeController:
    def __init__(self, driver: GuardedDriver, estop: LatchingEStop, watchdog: Watchdog,
                 proximity: ProximityMonitor | None = None, cfg: SafetyConfig = DEFAULT_SAFETY,
                 log: Callable[[str], None] | None = None):
        self._token = _Token()
        driver.bind(self._token)
        self.driver = driver
        self.estop = estop
        self.watchdog = watchdog
        self.proximity = proximity
        self.filter = SafetyFilter(cfg)
        self.cfg = cfg
        self.log = log or (lambda s: None)
        self.n_ticks = 0
        self.n_estop_checks = 0
        self.incidents: list[str] = []

    def freeze(self, reason: str) -> TickStatus:
        self.driver.freeze(self._token)
        return TickStatus(False, reason)

    def tick(self, q_target: np.ndarray, gripper_target: float, dt: float,
             monitor_rgb: np.ndarray | None = None, robot_moving: bool = True) -> TickStatus:
        self.n_ticks += 1
        self.n_estop_checks += 1
        if self.estop.check():
            self.log("ESTOP: tripped -> freeze")
            return self.freeze("estop")
        if not self.watchdog.ok():
            self.log("WATCHDOG: heartbeat stale -> freeze")
            return self.freeze("watchdog")
        if self.proximity is not None and self.proximity.update(monitor_rgb, robot_moving):
            self.log(f"PROXIMITY: {self.proximity.last_count} px person-like in workspace -> pause")
            return self.freeze("proximity")
        q_cur, g_cur = self.driver.read_state()
        cmd = self.filter.filter(q_target, gripper_target, q_cur, g_cur, dt)
        for e in cmd.events:
            if e.kind in ("workspace", "nan"):
                self.log(f"LIMIT: {e.kind}: {e.detail}")
        self.driver.send(self._token, cmd.q, cmd.gripper)
        return TickStatus(True, "held" if cmd.held else "ok", cmd.events)


def estop_file_default() -> Path:
    return Path(os.environ.get("TIDY_ESTOP_FILE", "/tmp/tidy_estop"))
