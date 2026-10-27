"""Training data shared by all policies: per-object-step segments of (obs, action) in the spec of policies/spec.py.

Sources: `teleop` (simulated leader-arm operator), `video` (retargeted pseudo-demos from human videos),
`video_raw` (ablation: raw hand trajectories), `practice` (auto-scored successful self-practice steps),
`simpre` (task-agnostic random pick-and-place in the digital twin, for pre-training).

Also exports datasets in a LeRobot-style layout (parquet episodes + meta/info.json) under data/robot/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from common.provenance import REPO_ROOT, sha256_obj
from policies.spec import ACT_DIM, DELTA_SCALE, OBS_DIM

# indices into the observation vector (policies/spec.py)
OBJ_REL = slice(8, 11)
GOAL_REL = slice(23, 26)
EE_POS = slice(0, 3)


@dataclass
class Segment:
    obs: np.ndarray       # (T, OBS_DIM)
    act: np.ndarray       # (T, ACT_DIM)
    source: str
    obj: str = ""
    meta: dict = field(default_factory=dict)


def dataset_hash(segs: list[Segment]) -> str:
    return sha256_obj([[s.obs, s.act, s.source] for s in segs])


def to_absolute(obs: np.ndarray) -> np.ndarray:
    """Ablation `-object-centric`: object/goal positions in the table frame instead of relative to the TCP."""
    o = obs.copy()
    ee = o[..., EE_POS] / 5.0
    o[..., OBJ_REL] = (o[..., OBJ_REL] / 10.0 + ee) * 5.0
    o[..., GOAL_REL] = (o[..., GOAL_REL] / 10.0 + ee) * 5.0
    return o


def drop_absolute(obs: np.ndarray) -> np.ndarray:
    """Remove the absolute TCP position (keeps only object-/goal-relative geometry), see D-021."""
    o = obs.copy()
    o[..., EE_POS] = 0.0
    return o


def minimal_obs(obs: np.ndarray) -> np.ndarray:
    """Only task-relative geometry + gripper state: drops absolute TCP position, jaw yaw and pitch (D-024)."""
    o = obs.copy()
    o[..., 0:6] = 0.0
    return o


def transform_obs(obs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "minimal":
        return minimal_obs(obs)
    if mode == "absolute":
        return to_absolute(obs)
    if mode == "rel_noabs":
        return drop_absolute(obs)
    return obs


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, min_std: float = 1e-2) -> Normalizer:
        """Per-feature statistics over axis 0 (chunks: pass (N, K, D) for per-offset statistics)."""
        return Normalizer(x.mean(axis=0), np.maximum(x.std(axis=0), min_std))

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inv(self, x):
        return x * self.std + self.mean

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_state(d: dict) -> Normalizer:
        return Normalizer(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))


def with_targets(s: Segment) -> Segment:
    """Attach the TCP position at each tick and the absolute TCP target of each action (computed from the raw
    observation, before any observation transform removes the TCP position)."""
    ee = s.obs[:, EE_POS] / 5.0
    tgt = ee + np.clip(s.act[:, :3], -1.0, 1.0) * DELTA_SCALE
    return Segment(s.obs, s.act, s.source, s.obj, {**s.meta, "ee": ee, "targets": tgt})


class ChunkDataset:
    """All (obs_t, act_{t..t+K-1}, mask) samples; chunks are padded with the last action (mask = 0).

    `cumulative=True` (D-025): the position part of chunk entry k is the TCP *target* of action t+k expressed
    relative to the TCP at time t (in DELTA_SCALE units), as in ACT's absolute-position chunks. Temporal
    ensembling then averages spatial targets. With per-tick deltas (`False`) it averages time-indexed
    increments, which drift whenever the arm lags the demonstrated speed."""

    def __init__(self, segs: list[Segment], K: int, weights: dict[str, float] | None = None,
                 cumulative: bool = False):
        obs, chunks, masks, w = [], [], [], []
        for s in segs:
            T = len(s.obs)
            if T == 0:
                continue
            pad = np.concatenate([s.act, np.repeat(s.act[-1:], K, axis=0)], axis=0)
            if cumulative:
                tg = np.asarray(s.meta["targets"])
                tpad = np.concatenate([tg, np.repeat(tg[-1:], K, axis=0)], axis=0)
                ee = np.asarray(s.meta["ee"])
            for t in range(min(T, int(s.meta.get("n_valid", T)))):   # synthetic segments: prefix only
                obs.append(s.obs[t])
                ch = pad[t:t + K].copy()
                if cumulative:
                    ch[:, :3] = (tpad[t:t + K] - ee[t]) / DELTA_SCALE
                chunks.append(ch)
                m = np.zeros(K, np.float32)
                m[: min(K, T - t)] = 1.0
                masks.append(m)
                w.append((weights or {}).get(s.source, 1.0))
        self.obs = np.array(obs, np.float32).reshape(-1, OBS_DIM)
        self.chunks = np.array(chunks, np.float32).reshape(-1, K, ACT_DIM)
        self.masks = np.array(masks, np.float32).reshape(-1, K)
        self.w = np.array(w, np.float64)
        self.K = K

    def __len__(self) -> int:
        return len(self.obs)

    def sampler(self, batch: int, rng: np.random.Generator, obs_norm: Normalizer, act_norm: Normalizer,
                aug_sigma: float = 0.0, device: str = "cpu"):
        p = self.w / self.w.sum()
        o_all = torch.tensor(obs_norm(self.obs), dtype=torch.float32)
        a_all = torch.tensor(act_norm(self.chunks), dtype=torch.float32)
        m_all = torch.tensor(self.masks)
        std = torch.tensor(obs_norm.std, dtype=torch.float32)
        while True:
            idx = torch.tensor(rng.choice(len(self), size=batch, p=p))
            o = o_all[idx].clone()
            if aug_sigma > 0:  # perception-noise augmentation on object/goal relative positions (metres)
                n = torch.randn(batch, 2, 3) * aug_sigma * 10.0
                n[..., 2] *= 0.3
                o[:, OBJ_REL] += n[:, 0] / std[OBJ_REL]
                o[:, GOAL_REL] += n[:, 1] / std[GOAL_REL]
            yield o.to(device), a_all[idx].to(device), m_all[idx].to(device)


# ------------------------------------------------------------------------------------------ LeRobot-style export
def export_lerobot(segs: list[Segment], name: str, fps: int = 10, extra_meta: dict | None = None) -> Path:
    """Write a LeRobot-style dataset: data/robot/<name>/data/chunk-000/episode_XXXXXX.parquet + meta/info.json.
    Columns follow LeRobot naming (observation.state, action, episode_index, frame_index, timestamp, index)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = REPO_ROOT / "data" / "robot" / name
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    gidx = 0
    episodes = []
    for e, s in enumerate(segs):
        T = len(s.obs)
        tbl = pa.table({
            "observation.state": [row.tolist() for row in s.obs],
            "action": [row.tolist() for row in s.act],
            "episode_index": [e] * T,
            "frame_index": list(range(T)),
            "timestamp": [t / fps for t in range(T)],
            "index": list(range(gidx, gidx + T)),
            "task_index": [0] * T,
        })
        pq.write_table(tbl, root / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
        episodes.append({"episode_index": e, "length": T, "source": s.source, "object": s.obj})
        gidx += T
    info = {"codebase_version": "v2.0-style (Tidy export; low-dim, no video)", "fps": fps,
            "total_episodes": len(segs), "total_frames": gidx, "robot_type": "so100-class (sim twin)",
            "features": {"observation.state": {"dtype": "float32", "shape": [OBS_DIM],
                                               "names": "see policies/spec.py"},
                         "action": {"dtype": "float32", "shape": [ACT_DIM], "names": "see policies/spec.py"}},
            "dataset_hash": dataset_hash(segs), **(extra_meta or {})}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=1))
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return root
