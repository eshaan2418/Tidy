import numpy as np

from policies.data import EE_POS, ChunkDataset, Segment, minimal_obs, with_targets
from policies.spec import ACT_DIM, DELTA_SCALE, OBS_DIM, View, obs_vector
from policies.train import TrainConfig, load_policy, train_policy
from sim.scene import LIB


def _seg(T=30, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(T, OBS_DIM)).astype(np.float32)
    obs[:, EE_POS] = np.cumsum(rng.normal(0, 0.01, (T, 3)), 0) * 5.0
    act = rng.uniform(-0.9, 0.9, size=(T, ACT_DIM)).astype(np.float32)
    return Segment(obs, act, "test")


def test_cumulative_chunks_are_targets_relative_to_tcp_at_t():
    s = with_targets(_seg())
    ds = ChunkDataset([s], K=5, cumulative=True)
    ee = s.obs[:, EE_POS] / 5.0
    t, k = 7, 3
    expect = (ee[t + k] + s.act[t + k, :3] * DELTA_SCALE - ee[t]) / DELTA_SCALE
    assert np.allclose(ds.chunks[t, k, :3], expect, atol=1e-5)
    assert np.allclose(ds.chunks[t, k, 3:], s.act[t + k, 3:])
    # k = 0 equals the plain per-tick delta
    assert np.allclose(ds.chunks[t, 0, :3], s.act[t, :3], atol=1e-5)


def test_synthetic_prefix_only():
    s = _seg()
    s.meta["n_valid"] = 4
    assert len(ChunkDataset([s], K=5)) == 4


def test_minimal_obs_drops_absolute_features():
    o = minimal_obs(_seg().obs)
    assert np.all(o[:, 0:6] == 0)


def test_cylinder_yaw_not_observed():
    v = View(np.zeros(3), 0.0, 0.0, 0.05, 0.0, np.array([0.2, 0, 0.03]), 1.0, LIB["cup"], np.zeros(3))
    o = obs_vector(v)
    assert np.all(o[11:13] == 0)
    v.obj = LIB["box"]
    assert np.any(obs_vector(v)[11:13] != 0)


def test_tiny_training_roundtrip_and_agent_targets():
    from policies.agent import PolicyAgent

    segs = [_seg(seed=i) for i in range(3)]
    pol = train_policy(segs, TrainConfig(steps=5, batch=16))
    pol2 = load_policy(pol.checkpoint_bytes())
    assert pol2.checkpoint_hash() == pol.checkpoint_hash()
    ag = PolicyAgent([pol2, pol2])
    v = View(np.array([0.15, 0.0, 0.12]), 0.0, 0.3, 0.05, 0.0, np.array([0.2, 0, 0.03]), 0.0, LIB["box"],
             np.array([0.04, 0.2, 0.045]))
    ag.begin_step(LIB["box"], v.obj_pos, 0.0, v.goal_pos)
    a = ag.act(v)
    step = np.asarray(a.pos) - v.ee_pos
    assert np.all(np.abs(step) <= DELTA_SCALE + 1e-9)   # bounded per-tick motion, before the safety filter
    assert ag.trace[-1]["phase"] == ["pick"] and len(ag.trace[-1]["disagree"]) == 1
