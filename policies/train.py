"""Train ACT / Diffusion Policy (single models or bootstrap ensembles) on Segment lists.

Compute budget is fixed per method (same steps / batch for every data source, Rule 6). Optional pre-training
on task-agnostic sim data, then fine-tuning with the *same* fine-tuning budget as from-scratch models.
Checkpoints go to checkpoints/ (git-ignored); their sha256 is recorded in every result file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from common.provenance import REPO_ROOT
from policies.data import ChunkDataset, Normalizer, Segment, dataset_hash, transform_obs, with_targets
from policies.models import ACT, DiffusionPolicy, MLPPolicy

CKPT_DIR = REPO_ROOT / "checkpoints"


@dataclass
class TrainConfig:
    kind: str = "act"            # act | dp | mlp
    K: int = 10                  # chunk length (ACT 10, DP 8)
    steps: int = 3000
    batch: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    aug_sigma: float = 0.0       # perception-noise augmentation (m); 0 chosen on validation cards (D-026)
    absolute_obs: bool = False   # ablation -object-centric
    act_mode: str = "cumulative"      # delta (per-tick increments) | cumulative (targets relative to TCP at t, D-025)
    obs_mode: str = "minimal"        # rel | rel_noabs (no absolute TCP position) | absolute
    seed: int = 0
    augment: int = 0             # >0: corrective keyframe augmentation, this many synthetic starts per phase (D-021)
    pretrain_steps: int = 0      # >0: pre-train on `pretrain` segments first (same fine-tune budget after)
    extra: dict = field(default_factory=dict)


@dataclass
class TrainedPolicy:
    cfg: TrainConfig
    model: torch.nn.Module
    obs_norm: Normalizer
    act_norm: Normalizer
    data_hash: str
    n_samples: int
    final_loss: float

    def checkpoint_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save({"cfg": asdict(self.cfg), "state": self.model.state_dict(), "obs_norm": self.obs_norm.state(),
                    "act_norm": self.act_norm.state(), "data_hash": self.data_hash, "n_samples": self.n_samples,
                    "final_loss": self.final_loss}, buf)
        return buf.getvalue()

    def checkpoint_hash(self) -> str:
        return hashlib.sha256(self.checkpoint_bytes()).hexdigest()

    def save(self, name: str) -> str:
        CKPT_DIR.mkdir(exist_ok=True)
        b = self.checkpoint_bytes()
        (CKPT_DIR / f"{name}.pt").write_bytes(b)
        return hashlib.sha256(b).hexdigest()


def make_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.kind == "act":
        return ACT(K=cfg.K)
    if cfg.kind == "mlp":
        return MLPPolicy(K=cfg.K)
    return DiffusionPolicy(K=cfg.K)


def obs_mode(cfg: TrainConfig) -> str:
    return "absolute" if cfg.absolute_obs else cfg.obs_mode


def _prep(segs: list[Segment], mode: str) -> list[Segment]:
    segs = [with_targets(s) for s in segs]
    if mode == "rel":
        return segs
    return [Segment(transform_obs(s.obs, mode), s.act, s.source, s.obj, s.meta) for s in segs]


def _fit(model, ds: ChunkDataset, cfg: TrainConfig, steps: int, on: Normalizer, an: Normalizer,
         rng: np.random.Generator) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(steps, 1), pct_start=0.05)
    it = ds.sampler(cfg.batch, rng, on, an, cfg.aug_sigma)
    model.train()
    ema = None
    for _ in range(steps):
        o, a, m = next(it)
        loss = model.loss(o, a, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        lv = float(loss.detach())
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
    model.eval()
    return float(ema or 0.0)


def train_policy(segs: list[Segment], cfg: TrainConfig, pretrain: list[Segment] | None = None) -> TrainedPolicy:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data_hash = dataset_hash(segs)          # hash of the human-provided data, before augmentation
    if cfg.augment > 0:
        from policies.augment import augment

        segs = augment(segs, cfg.augment, seed=cfg.seed)
        if pretrain:
            pretrain = augment(pretrain, max(1, cfg.augment // 2), seed=cfg.seed + 1)
    segs = _prep(segs, obs_mode(cfg))
    cum = cfg.act_mode == "cumulative"
    ds = ChunkDataset(segs, cfg.K, cumulative=cum)
    if len(ds) == 0:
        raise ValueError("no training samples")
    # normalisers from the target data (pre-training reuses them so fine-tuning sees a consistent scale)
    on = Normalizer.fit(ds.obs)
    an = Normalizer.fit(ds.chunks) if cum else Normalizer.fit(ds.chunks.reshape(-1, ds.chunks.shape[-1]))
    model = make_model(cfg)
    if pretrain and cfg.pretrain_steps > 0:
        pds = ChunkDataset(_prep(pretrain, obs_mode(cfg)), cfg.K, cumulative=cum)
        _fit(model, pds, cfg, cfg.pretrain_steps, on, an, rng)
    loss = _fit(model, ds, cfg, cfg.steps, on, an, rng)
    return TrainedPolicy(cfg, model, on, an, data_hash, len(ds), loss)


def bootstrap(segs: list[Segment], rng: np.random.Generator) -> list[Segment]:
    idx = rng.integers(0, len(segs), len(segs))
    return [segs[i] for i in idx]


def train_ensemble(segs: list[Segment], cfg: TrainConfig, n: int, pretrain: list[Segment] | None = None,
                   bootstrap_data: bool = True) -> list[TrainedPolicy]:
    """Member i: seed cfg.seed*100+i, trained on a bootstrap resample of the segments (member 0: all data)."""
    out = []
    for i in range(n):
        c = TrainConfig(**{**asdict(cfg), "seed": cfg.seed * 100 + i})
        rng = np.random.default_rng(c.seed)
        data = segs if (i == 0 or not bootstrap_data) else bootstrap(segs, rng)
        out.append(train_policy(data, c, pretrain))
    return out


def _train_job(args) -> bytes:
    segs, cfg_d, pretrain = args
    torch.set_num_threads(2)
    return train_policy(segs, TrainConfig(**cfg_d), pretrain).checkpoint_bytes()


def load_policy(b: bytes) -> TrainedPolicy:
    d = torch.load(io.BytesIO(b), weights_only=False)
    cfg = TrainConfig(**d["cfg"])
    m = make_model(cfg)
    m.load_state_dict(d["state"])
    m.eval()
    return TrainedPolicy(cfg, m, Normalizer.from_state(d["obs_norm"]), Normalizer.from_state(d["act_norm"]),
                         d["data_hash"], int(d.get("n_samples", 0)), float(d.get("final_loss", 0.0)))


def train_many(jobs: list[tuple[list[Segment], TrainConfig, list[Segment] | None]],
               workers: int | None = None) -> list[TrainedPolicy]:
    """Train independent policies in parallel processes (2 torch threads each). Deterministic per job seed."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = workers or max(1, min(len(jobs), ((os.cpu_count() or 4) - 2) // 2))
    args = [(s, asdict(c), p) for s, c, p in jobs]
    if workers == 1 or len(jobs) == 1:
        return [load_policy(_train_job(a)) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return [load_policy(b) for b in ex.map(_train_job, args)]


def ensemble_jobs(segs: list[Segment], cfg: TrainConfig, n: int, pretrain: list[Segment] | None = None,
                  bootstrap_data: bool = True) -> list[tuple[list[Segment], TrainConfig, list[Segment] | None]]:
    """Jobs for train_many equivalent to train_ensemble (member i: seed cfg.seed*100+i, bootstrap resample)."""
    out = []
    for i in range(n):
        c = TrainConfig(**{**asdict(cfg), "seed": cfg.seed * 100 + i})
        data = segs if (i == 0 or not bootstrap_data) else bootstrap(segs, np.random.default_rng(c.seed))
        out.append((data, c, pretrain))
    return out


def ensemble_hash(pols: list[TrainedPolicy]) -> str:
    h = hashlib.sha256()
    for p in pols:
        h.update(p.checkpoint_hash().encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a policy for a chore from processed human videos (sim stage).")
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--kind", default="act", choices=["act", "dp"])
    ap.add_argument("--n-videos", type=int, default=10)
    args = ap.parse_args()
    from bench.common import video_segments

    segs, _ = video_segments(args.chore, args.n_videos)
    pol = train_policy(segs, TrainConfig(kind=args.kind, K=10 if args.kind == "act" else 8))
    h = pol.save(f"{args.chore}_{args.kind}_v{args.n_videos}")
    print(f"trained {args.kind} on {pol.n_samples} samples, loss {pol.final_loss:.4f}, checkpoint sha256 {h}")


if __name__ == "__main__":
    main()
