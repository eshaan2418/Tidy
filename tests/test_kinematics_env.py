import numpy as np
import pytest

from common.kinematics import GEOM, fk, ik
from sim.env import Placement, TidyEnv
from sim.expert import PickPlaceTask, ScriptedPickPlace
from sim.scene import FIXTURES, LIB, MAT_X, MAT_Y


def test_ik_fk_roundtrip_over_mat():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = np.array([rng.uniform(*MAT_X), rng.uniform(*MAT_Y), rng.uniform(0.015, 0.10)])
        psi = rng.uniform(-np.pi / 2, np.pi / 2)
        r = ik(p, psi, 0.0)
        assert r.pos_error == 0.0, p
        pos, psi2, pitch = fk(r.q)
        assert np.allclose(pos, p, atol=1e-6)
        assert abs(((psi2 - psi + np.pi / 2) % np.pi) - np.pi / 2) < 1e-6
        assert np.all(r.q >= np.array(GEOM.q_min)) and np.all(r.q <= np.array(GEOM.q_max))


def test_mujoco_tcp_matches_analytic_fk():
    env = TidyEnv([LIB["cup"]], [FIXTURES["bin"]])
    env.reset(Placement({"cup": (0.2, 0.0, 0.0)}))
    q = ik(np.array([0.18, 0.05, 0.06]), 0.4, 0.2).q
    env.data.qpos[env._jq] = q
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    p, _ = env.tcp_pose()
    assert np.allclose(p, fk(q)[0], atol=1e-4)


@pytest.mark.slow
def test_scripted_pick_place_smoke():
    """Env sanity: the scripted controller clears a 4-object desk into the bin."""
    objs = [LIB[n] for n in ["pen", "cup", "tape", "box"]]
    env = TidyEnv(objs, [FIXTURES["bin"]])
    env.reset(Placement({"pen": (0.14, -0.08, 0.4), "cup": (0.22, 0.05, 0), "tape": (0.25, -0.10, 0),
                         "box": (0.12, 0.08, 1.0)}))
    fb = FIXTURES["bin"]
    for o in objs:
        p, y = env.obj_pose(o.name)
        ctl = ScriptedPickPlace(PickPlaceTask(o, np.array([fb.pos[0], fb.pos[1], 0.0])), p, y)
        for _ in range(250):
            ctl_act = ctl.act(env.ee_state()[0])
            env.step(ctl_act)
            if ctl.done:
                break
        env.settle(100)
    assert all(fb.contains_xy(env.obj_pose(o.name)[0]) for o in objs)
    assert not [e for e in env.safety_events if not e.startswith("held")]
