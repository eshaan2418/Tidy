"""Policy design ablation on VALIDATION cards (`pool` split, never eval): the evidence behind D-021..D-026.

Data: desk clearing, 10 videos (seed 0) and teleop at equal human time (seed 0). One training seed per cell, 30
validation trials per cell, so treat the differences as indicative. The headline experiments (M4-M7) use only the
configuration chosen here (ACT, cumulative chunks, minimal obs, no augmentation).
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from bench.common import (
    Condition,
    ckey,
    cmdline,
    evaluate,
    object_rate,
    program_from_records,
    rate,
    segments_from_records,
    train_conditions,
    video_records,
)
from bench.m4_video_vs_teleop import teleop_condition, video_condition
from common.provenance import write_result
from eval.placements import training_cards
from policies.train import TrainConfig

VARIANTS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "delta_rel": (dict(act_mode="delta", obs_mode="rel"), {}),
    "delta_rel_newest": (dict(act_mode="delta", obs_mode="rel"), {"temporal_m": -3.0}),
    "cumulative_rel": (dict(act_mode="cumulative", obs_mode="rel"), {}),
    "cumulative_minimal (default)": (dict(), {}),
    "default + obs-noise aug 4mm": (dict(aug_sigma=0.004), {}),
    "default + corrective aug": (dict(augment=6), {}),
    "default, K=20": (dict(K=20), {}),
    "default, 6000 steps": (dict(steps=6000), {}),
    "mlp (cumulative, minimal)": (dict(kind="mlp"), {}),
    "diffusion policy (cumulative, minimal)": (dict(kind="dp", K=8), {}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--n-val", type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()
    v = video_condition(args.chore, 10, 0)
    t = teleop_condition(args.chore, v.ledger.total_s(), 0, "teleop_1x")
    recs = video_records(args.chore, training_cards(args.chore, "film", 10))
    no_relabel = segments_from_records(recs, "video", None)
    conds = []
    for name, (cfg, kw) in VARIANTS.items():
        for src in (v, t):
            conds.append(Condition(f"{src.name}|{name}", args.chore, 0, src.segs, src.program, src.ledger,
                                   TrainConfig(seed=0, **cfg), agent_kw=kw))
    conds.append(Condition("video_10|default, no goal relabelling", args.chore, 0, no_relabel,
                           program_from_records(args.chore, recs), v.ledger, TrainConfig(seed=0)))
    pols = train_conditions(conds)
    val = training_cards(args.chore, "pool", args.n_val)
    rows: dict[str, Any] = {}
    for c in conds:
        tr = evaluate(args.chore, pols[ckey(c)], c.program, "val", cards=val, agent_kw=c.agent_kw)
        rows[c.name] = {"success": rate(tr), "objects": object_rate(tr), "n_segments": len(c.segs)}
        print(c.name, rows[c.name]["success"]["k"], "/", rows[c.name]["success"]["n"],
              round(rows[c.name]["objects"]["p"], 2), flush=True)
    write_result(f"sim/dev_policy_design_{args.chore}", {"rows": rows, "wall_s": time.time() - t0}, stage="sim",
                 config={"variants": {k: v[0] | {"agent": v[1]} for k, v in VARIANTS.items()}, "n_val": args.n_val,
                         "split": "pool (validation)"}, seed=0, command=cmdline())


if __name__ == "__main__":
    main()
