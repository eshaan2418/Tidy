"""M2 + M4 core experiment (sim stage): learning from ~10 human videos vs teleoperation at equal human time,
learning curves over human minutes, and the M4 ablations.

Conditions (each repeated for `--seeds` independent data collections + trainings; same policy class, config and
compute for every method, Rule 6):
  video_N         N phone videos (full pipeline: perception -> phases -> retargeting -> object-centric targets)
  teleop_xB       simulated leader-arm teleop with B x the human-time budget of video_10
  teleop_fast_1x  sensitivity: a faster/more precise operator at 1x budget
  video_raw       ablation -retargeting (raw wrist trajectories, human timing)
  video_hand      ablation -object-centric (hand-only grasp/place targets, no snapping to tracked objects)
  video_simpre    + task-agnostic sim pre-training (D-019), same fine-tuning budget
  video_practice  + one round of supervised self-practice (human supervision + resets are charged)
Evaluation: frozen eval_seen (40) and eval_unseen_place (40) cards per seed.
"""

from __future__ import annotations

import argparse
import time

from bench.common import (
    Condition,
    HumanLedger,
    ckey,
    cmdline,
    evaluate_condition,
    ledger_from_records,
    pooled,
    practice_data,
    program_from_records,
    segments_from_records,
    simpre_segments,
    strip_trials,
    teleop_data,
    train_conditions,
    video_records,
)
from common.provenance import write_result
from eval.placements import training_cards
from policies.train import TrainConfig


def video_condition(chore: str, n: int, seed: int, variant: str = "full", name: str | None = None) -> Condition:
    cards = training_cards(chore, "film", n, offset=100 * seed)
    recs = video_records(chore, cards, variant)
    prog = program_from_records(chore, recs)
    segs = segments_from_records(recs, "video" if variant == "full" else f"video_{variant}", prog)
    acc = sum(r.accepted for r in recs)
    return Condition(name or f"video_{n}", chore, seed, segs, prog, ledger_from_records(recs), TrainConfig(seed=seed),
                     meta={"n_videos": n, "accepted": acc, "variant": variant})


def teleop_condition(chore: str, budget_s: float, seed: int, name: str, operator: str = "default") -> Condition:
    segs, L, eps, prog = teleop_data(chore, budget_s, seed=seed, operator=operator)
    return Condition(name, chore, seed, segs, prog, L, TrainConfig(seed=seed),
                     meta={"n_demos": len(eps), "operator_success": sum(e.success for e in eps), "budget_s": budget_s})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 seed, 10 trials per split, core conditions only")
    args = ap.parse_args()
    seeds = [0] if args.quick else list(range(args.seeds))
    n_eval = 10 if args.quick else None
    t0 = time.time()
    conds: list[Condition] = []
    pre = [] if args.quick else simpre_segments(120)
    for s in seeds:
        v10 = video_condition(args.chore, 10, s)
        budget = v10.ledger.total_s()
        conds += [v10, teleop_condition(args.chore, budget, s, "teleop_1x")]
        if not args.quick:
            conds += [video_condition(args.chore, 5, s), video_condition(args.chore, 20, s),
                      teleop_condition(args.chore, 2 * budget, s, "teleop_2x"),
                      teleop_condition(args.chore, 4 * budget, s, "teleop_4x"),
                      teleop_condition(args.chore, budget, s, "teleop_fast_1x", operator="fast"),
                      video_condition(args.chore, 10, s, "raw", "video_raw"),
                      video_condition(args.chore, 10, s, "hand", "video_hand")]
            c = video_condition(args.chore, 10, s, name="video_simpre")
            c.pretrain = pre
            c.cfg = TrainConfig(seed=s, pretrain_steps=3000)
            c.meta["pretrain_segments"] = len(pre)
            conds.append(c)
    pols = train_conditions(conds)
    # self-practice: start from the trained video_10 policy of each seed, practice, retrain on video + practice
    if not args.quick:
        prac = []
        for c in [c for c in conds if c.name == "video_10"]:
            assert c.program is not None
            psegs, L, stats = practice_data(args.chore, pols[ckey(c)], c.program, 20, seed=c.seed,
                                            ledger=HumanLedger(list(c.ledger.entries)))
            prac.append(Condition("video_practice", args.chore, c.seed, c.segs + psegs, c.program, L,
                                  TrainConfig(seed=c.seed), meta={**c.meta, "practice": stats}))
        pols.update(train_conditions(prac))
        conds += prac
    results = []
    for c in conds:
        r = evaluate_condition(c, pols[ckey(c)], n=n_eval)
        results.append(r)
        brief = {sp: (d["success"]["k"], d["success"]["n"], round(d["objects"]["p"], 2))
                 for sp, d in r["splits"].items()}
        print(c.name, c.seed, r["human"]["total_min"], "min |", brief, flush=True)
    names = list(dict.fromkeys(c.name for c in conds))
    summary = {}
    for nm in names:
        rs = [r for r in results if r["name"] == nm]
        summary[nm] = {sp: {"success": pooled(rs, sp), "objects": pooled(rs, sp, "objects")}
                       for sp in rs[0]["splits"]}
    out = write_result(f"sim/m4_video_vs_teleop_{args.chore}" + ("_quick" if args.quick else ""),
                       {"summary": summary, "runs": [strip_trials(r) for r in results],
                        "wall_s": time.time() - t0},
                       stage="sim", config={"chore": args.chore, "seeds": seeds, "n_eval": n_eval,
                                            "train": TrainConfig().__dict__},
                       seed=seeds[0], checkpoint_hash=None, dataset_hash=None, command=cmdline())
    print("wrote", out)


if __name__ == "__main__":
    main()
