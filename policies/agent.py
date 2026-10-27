"""Runtime agent: wraps trained ACT / DP policies (single or ensemble) behind the executor's Agent protocol.

ACT: a chunk is predicted every tick and executed with temporal ensembling (exponential weights, m = 0.1).
DP: a chunk is sampled and the first `horizon` actions are executed (receding horizon).
Uncertainty signals recorded per tick (used by active/ and recovery/):
  * `disagree`: std across ensemble members of the predicted next TCP delta (m)
  * `spread`: DP only, std across S action samples of the next TCP delta (m) (action-distribution entropy proxy)
The safety filter and IK sit below this agent (D-006): it can never command a joint outside the limits.
"""

from __future__ import annotations

import numpy as np
import torch

from common.kinematics import preferred_pitch
from policies.data import transform_obs
from policies.spec import DELTA_SCALE, View, action_from_vector, obs_vector
from policies.train import TrainedPolicy, obs_mode
from sim.env import EEAction
from sim.scene import ObjectSpec


class PolicyAgent:
    def __init__(self, policies: list[TrainedPolicy], temporal_m: float = 0.1, dp_horizon: int = 4,
                 dp_samples: int = 1, seed: int = 0, kin_pitch: bool | None = None):
        assert policies
        self.pols = policies
        self.kind = policies[0].cfg.kind
        self.obs_mode = obs_mode(policies[0].cfg)
        self.cumulative = policies[0].cfg.act_mode == "cumulative"
        self.kin_pitch = (self.obs_mode == "minimal") if kin_pitch is None else kin_pitch
        self.m = temporal_m
        self.dp_horizon = dp_horizon
        self.dp_samples = dp_samples
        self.gen = torch.Generator().manual_seed(seed)
        self.trace: list[dict] = []
        self._hist: list[tuple[int, np.ndarray]] = []
        self._queue: list[np.ndarray] = []
        self._t = 0

    def begin_step(self, obj: ObjectSpec, obj_pos: np.ndarray, obj_yaw: float, goal: np.ndarray) -> None:
        self._hist, self._queue, self._t = [], [], 0
        self.trace.append({"obj": obj.name, "disagree": [], "spread": []})

    def _chunks(self, o: np.ndarray) -> np.ndarray:
        """(members, K, ACT_DIM) de-normalised chunks."""
        outs = []
        for p in self.pols:
            x = torch.tensor(p.obs_norm(o)[None], dtype=torch.float32)
            if self.kind in ("act", "mlp"):
                y = p.model.predict(x)[0].numpy()  # type: ignore[operator]
            else:
                xs = x.repeat(max(1, self.dp_samples), 1)
                y = p.model.sample(xs, generator=self.gen).numpy()  # type: ignore[operator]
                if self.dp_samples > 1:
                    d = p.act_norm.inv(y)[:, 0, :3] * DELTA_SCALE
                    self.trace[-1]["spread"].append(float(np.linalg.norm(d.std(axis=0))))
                y = y[0]
            outs.append(p.act_norm.inv(y))
        return np.array(outs)

    def _to_abs(self, chunk: np.ndarray, ee: np.ndarray) -> np.ndarray:
        """Cumulative chunks: positions relative to the TCP at prediction time -> absolute table-frame targets."""
        if not self.cumulative:
            return chunk
        c = chunk.copy()
        c[:, :3] = ee + c[:, :3] * DELTA_SCALE
        return c

    def act(self, view: View) -> EEAction:
        o = obs_vector(view)
        o = transform_obs(o, self.obs_mode)
        if self.kind in ("act", "mlp"):
            ch = self._chunks(o)
            if len(ch) > 1:
                self.trace[-1]["disagree"].append(float(np.linalg.norm(ch[:, 0, :3].std(axis=0)) * DELTA_SCALE))
            mean = self._to_abs(ch.mean(axis=0), view.ee_pos)
            self._hist.append((self._t, mean))
            K = mean.shape[0]
            self._hist = [(t0, c) for t0, c in self._hist if self._t - t0 < K]
            acts = np.array([c[self._t - t0] for t0, c in self._hist])
            w = np.exp(-self.m * np.arange(len(acts)))      # oldest prediction gets the largest weight (ACT)
            a = (w[:, None] * acts).sum(0) / w.sum()
        else:
            if not self._queue:
                ch = self._chunks(o)
                if len(ch) > 1:
                    self.trace[-1]["disagree"].append(
                        float(np.linalg.norm(ch[:, 0, :3].std(axis=0)) * DELTA_SCALE))
                self._queue = list(self._to_abs(ch.mean(axis=0), view.ee_pos)[: self.dp_horizon])
            a = self._queue.pop(0)
        self._t += 1
        if self.cumulative:  # averaged absolute target -> increment from the current TCP
            a = a.copy()
            a[:3] = (a[:3] - view.ee_pos) / DELTA_SCALE
        act = action_from_vector(view.ee_pos, a)
        if self.kin_pitch:  # pitch from the arm's kinematics (most top-down reachable), not learned
            act.pitch = preferred_pitch(act.pos)
        return act
