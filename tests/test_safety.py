"""Safety tests. These must never be skipped (marker: safety)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from common.kinematics import HOME_Q, fk
from safety.core import (
    FileEStop,
    GuardedDriver,
    LatchingEStop,
    ManualEStop,
    ProximityMonitor,
    SafeController,
    SafetyFilter,
    Watchdog,
)
from safety.limits import DEFAULT_SAFETY, in_no_go

pytestmark = pytest.mark.safety
ROOT = Path(__file__).resolve().parents[1]
CFG = DEFAULT_SAFETY


class FakeDriver(GuardedDriver):
    def __init__(self, q=None):
        super().__init__()
        self.q = np.array(HOME_Q if q is None else q, dtype=float)
        self.g = 0.0
        self.sent: list[np.ndarray] = []
        self.frozen = 0

    def read_state(self):
        return self.q.copy(), self.g

    def send(self, token, q, gripper):
        self._check(token)
        self.q = np.asarray(q, dtype=float).copy()
        self.g = gripper
        self.sent.append(self.q.copy())

    def freeze(self, token):
        self._check(token)
        self.frozen += 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_ctrl(driver=None, clock=None, proximity=None):
    driver = driver or FakeDriver()
    clock = clock or Clock()
    btn = ManualEStop()
    ctrl = SafeController(driver, LatchingEStop([btn]), Watchdog(CFG.watchdog_timeout_s, clock), proximity, CFG)
    return ctrl, driver, btn, clock


def test_limits_are_conservative():
    assert CFG.max_tcp_speed <= 0.25
    assert max(CFG.max_joint_vel) <= 3.0
    assert CFG.max_effort_frac <= 0.6
    assert CFG.joint_margin >= 0.05
    assert CFG.workspace.lo[2] > 0.0  # never command the TCP into the table
    assert CFG.watchdog_timeout_s <= 1.0


@pytest.mark.parametrize("seed", range(5))
def test_random_commands_never_exceed_limits(seed):
    """Fuzz: arbitrary (including adversarial / NaN / huge) targets never violate any limit."""
    rng = np.random.default_rng(seed)
    ctrl, drv, _, clock = make_ctrl()
    dt = 0.1
    for _ in range(400):
        mode = rng.integers(4)
        if mode == 0:
            q_t = rng.uniform(-6, 6, 5)
        elif mode == 1:
            q_t = drv.q + rng.normal(0, 0.5, 5)
        elif mode == 2:
            q_t = np.full(5, np.nan)
        else:
            q_t = rng.uniform(-1e6, 1e6, 5)
        q_prev = drv.q.copy()
        p_prev = fk(q_prev)[0]
        ctrl.watchdog.heartbeat()
        clock.t += dt
        ctrl.watchdog.heartbeat()
        ctrl.tick(q_t, float(rng.uniform(-3, 3)), dt)
        q = drv.q
        assert np.all(np.isfinite(q))
        assert np.all(q >= CFG.q_lo - 1e-9) and np.all(q <= CFG.q_hi + 1e-9)
        assert np.all(np.abs(q - q_prev) <= np.array(CFG.max_joint_vel) * dt + 1e-9)
        p = fk(q)[0]
        assert np.linalg.norm(p - p_prev) <= CFG.max_tcp_speed * dt + 1e-6
        assert CFG.workspace.contains(p, margin=-1e-9)
        assert not in_no_go(p)
        assert 0.0 <= drv.g <= 1.0


def test_estop_freezes_and_latches():
    ctrl, drv, btn, _ = make_ctrl()
    ctrl.tick(HOME_Q + 0.01, 0.0, 0.1)
    n = len(drv.sent)
    btn.pressed = True
    st = ctrl.tick(HOME_Q + 0.05, 0.0, 0.1)
    assert not st.sent and st.reason == "estop" and drv.frozen == 1
    btn.pressed = False  # releasing the button does NOT un-latch
    st = ctrl.tick(HOME_Q + 0.05, 0.0, 0.1)
    assert not st.sent and len(drv.sent) == n
    assert not ctrl.estop.reset("ok")  # wrong phrase
    assert ctrl.estop.reset(LatchingEStop.RESET_PHRASE)
    assert ctrl.tick(HOME_Q, 0.0, 0.1).sent


def test_estop_reset_refused_while_pressed(tmp_path):
    f = tmp_path / "stop"
    es = LatchingEStop([FileEStop(f)])
    f.touch()
    assert es.check()
    assert not es.reset(LatchingEStop.RESET_PHRASE)
    f.unlink()
    assert es.reset(LatchingEStop.RESET_PHRASE)


def test_estop_requires_a_source():
    with pytest.raises(ValueError):
        LatchingEStop([])


def test_watchdog_freezes_on_stale_heartbeat():
    ctrl, drv, _, clock = make_ctrl()
    clock.t = 0.0
    ctrl.watchdog.heartbeat()
    assert ctrl.tick(HOME_Q, 0, 0.1).sent
    clock.t = CFG.watchdog_timeout_s + 0.01
    st = ctrl.tick(HOME_Q, 0, 0.1)
    assert not st.sent and st.reason == "watchdog" and drv.frozen == 1


def test_proximity_pauses_and_fails_closed():
    clock = Clock()
    prox = ProximityMonitor(CFG, roi=None, clock=clock)
    ctrl, drv, _, _ = make_ctrl(clock=clock, proximity=prox)
    clear = np.full((60, 80, 3), 200, np.uint8)  # grey, no skin
    hand = clear.copy()
    hand[10:40, 10:40] = (205, 160, 130)  # skin-coloured patch
    assert ctrl.tick(HOME_Q, 0, 0.1, monitor_rgb=clear).sent
    assert not ctrl.tick(HOME_Q, 0, 0.1, monitor_rgb=hand).sent
    # stays paused until clear for proximity_resume_clear_s
    assert not ctrl.tick(HOME_Q, 0, 0.1, monitor_rgb=clear).sent
    clock.t += CFG.proximity_resume_clear_s + 0.1
    ctrl.watchdog.heartbeat()
    assert ctrl.tick(HOME_Q, 0, 0.1, monitor_rgb=clear).sent
    # missing camera frame -> pause (fail closed)
    assert not ctrl.tick(HOME_Q, 0, 0.1, monitor_rgb=None).sent


def test_driver_rejects_commands_bypassing_controller():
    drv = FakeDriver()
    with pytest.raises(PermissionError):
        drv.send(object(), HOME_Q, 0.0)
    ctrl, _, _, _ = make_ctrl(driver=drv)
    with pytest.raises(PermissionError):
        drv.send(object(), HOME_Q, 0.0)
    with pytest.raises(RuntimeError):
        make_ctrl(driver=drv)  # cannot rebind to a second (unchecked) controller


def test_filter_rejects_table_collision_pose():
    f = SafetyFilter(CFG)
    q_cur = HOME_Q.copy()
    q_bad = q_cur.copy()
    q_bad[1] = 1.6  # throw the shoulder forward/down
    for _ in range(50):
        out = f.filter(q_bad, 0.0, q_cur, 0.0, 0.1)
        q_cur = out.q
        ok, _ = f.pose_ok(q_cur)
        assert ok


def _calls_in(fn_node: ast.AST) -> set[str]:
    names = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def test_estop_path_present_in_controller_tick():
    """Fails if someone removes the e-stop / watchdog / filter checks from SafeController.tick."""
    tree = ast.parse((ROOT / "safety" / "core.py").read_text())
    tick = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "tick")
    calls = _calls_in(tick)
    assert {"check", "ok", "filter", "freeze", "send"} <= calls
    # e-stop check must come before any send
    src = ast.unparse(tick)
    assert src.index("self.estop.check()") < src.index("self.driver.send(")


def test_supervised_session_uses_safe_controller():
    """Real robot only via scripts/supervised_session.py, which must build a SafeController,
    check operator presence, and never construct a driver without it."""
    p = ROOT / "scripts" / "supervised_session.py"
    src = p.read_text()
    tree = ast.parse(src)
    calls = set()
    for n in ast.walk(tree):
        calls |= _calls_in(n)
    assert "SafeController" in calls
    assert "confirm_operator_presence" in calls
    assert "LatchingEStop" in calls
    assert "require_safety_log_resolved" in calls


def test_no_other_module_constructs_real_driver():
    offenders = []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT).as_posix()
        if rel.startswith((".venv", "tests/", "hardware/so100_driver.py", "scripts/supervised_session.py")):
            continue
        if "SO100Driver(" in py.read_text():
            offenders.append(rel)
    assert not offenders, f"real driver constructed outside supervised_session: {offenders}"
