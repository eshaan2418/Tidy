"""M4 supplement (sim): the teleop baseline with its best validated configuration (corrective augmentation on,
D-021), at 1x / 2x / 4x the human time of video_10. Same data, seeds and evaluation cards as
bench/m4_video_vs_teleop.py; only the training configuration of the teleop policies differs.
"""

from __future__ import annotations

import argparse
import time

from bench.common import ckey, cmdline, evaluate_condition, pooled, strip_trials, train_conditions
from bench.m4_video_vs_teleop import teleop_condition, video_condition
from common.provenance import write_result
from policies.train import TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    seeds = [0] if a.quick else list(range(a.seeds))
    conds = []
    for s in seeds:
        budget = video_condition(a.chore, 10, s).ledger.total_s()
        for mult in ((1,) if a.quick else (1, 2, 4)):
            c = teleop_condition(a.chore, mult * budget, s, f"teleop_{mult}x_aug")
            c.cfg = TrainConfig(seed=s, augment=6)
            conds.append(c)
    pols = train_conditions(conds)
    results = []
    for c in conds:
        r = evaluate_condition(c, pols[ckey(c)], n=10 if a.quick else None)
        results.append(r)
        print(c.name, c.seed, {sp: (d["success"]["k"], d["success"]["n"]) for sp, d in r["splits"].items()},
              flush=True)
    summary = {}
    for nm in dict.fromkeys(c.name for c in conds):
        rs = [r for r in results if r["name"] == nm]
        summary[nm] = {sp: {"success": pooled(rs, sp), "objects": pooled(rs, sp, "objects")} for sp in rs[0]["splits"]}
    write_result(f"sim/m4_teleop_augmented_{a.chore}" + ("_quick" if a.quick else ""),
                 {"summary": summary, "runs": [strip_trials(r) for r in results], "wall_s": time.time() - t0},
                 stage="sim", config={"chore": a.chore, "seeds": seeds, "train": TrainConfig(augment=6).__dict__},
                 seed=seeds[0], command=cmdline())


if __name__ == "__main__":
    main()
