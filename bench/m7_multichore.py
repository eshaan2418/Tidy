"""M7 (sim stage): chores 2-4 + the held-out chore from ~10 videos each, and the generalization matrix.

Part A, per chore: video_10 vs teleop_1x (equal human time), 3 seeds, eval_seen + eval_unseen_place (+ unseen
objects where the chore defines them). Held-out chore (`tray_heldout`): taught only via 10 videos at test time,
(a) from scratch and (b) after multi-chore pre-training on the video pseudo-demos of chores 1-4 (no tray data).
Part B, desk clearing (M4 checkpoints): unseen objects, lighting variants, mild clutter (2 distractors of new
colours / of the same colours as task objects), camera bump (1 cm / 2 cm extrinsic error, no recalibration).
Towel folding (chore 5) is not simulated (no deformable model in the twin): reported as not run.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from bench.common import (
    Condition,
    ckey,
    cmdline,
    evaluate,
    evaluate_condition,
    object_rate,
    pooled,
    program_from_records,
    rate,
    segments_from_records,
    strip_trials,
    train_conditions,
    video_records,
)
from bench.m4_video_vs_teleop import teleop_condition, video_condition
from bench.m6_recovery import load_ckpts
from common.provenance import write_result
from eval.chores import CHORES
from eval.placements import add_clutter, load_protocol, training_cards
from policies.train import TrainConfig, ensemble_hash

CHORES_M7 = ("sorting", "shelf_putaway", "stacking", "tray_heldout")
PRIMARY = ("desk_clearing", "sorting", "shelf_putaway", "stacking")


def part_a(seeds: list[int], n_eval: int | None) -> dict:
    conds: list[Condition] = []
    for chore in CHORES_M7:
        for s in seeds:
            v = video_condition(chore, 10, s)
            conds += [v, teleop_condition(chore, v.ledger.total_s(), s, "teleop_1x")]
            if chore == "tray_heldout":
                pre = []
                for other in PRIMARY:
                    recs = video_records(other, training_cards(other, "film", 10, offset=100 * s))
                    pre += segments_from_records(recs, "video_other", program_from_records(other, recs))
                c = video_condition(chore, 10, s, name="video_10_multipre")
                c.pretrain = pre
                c.cfg = TrainConfig(seed=s, pretrain_steps=3000)
                c.meta["pretrain_segments"] = len(pre)
                conds.append(c)
    pols = train_conditions(conds)
    results: list[dict[str, Any]] = []
    for c in conds:
        splits = ["eval_seen", "eval_unseen_place"] + (["eval_unseen_obj"] if CHORES[c.chore].novel_objects else [])
        r = evaluate_condition(c, pols[ckey(c)], splits=tuple(splits), n=n_eval)
        succ_ticks = [t["ticks"] for sp in r["splits"].values() for t in sp["trials"] if t["success"]]
        r["time_per_chore_s"] = float(np.mean(succ_ticks) * 0.1) if succ_ticks else None
        results.append(strip_trials(r))
        print(c.chore, c.name, c.seed, {sp: (d["success"]["k"], d["success"]["n"]) for sp, d in r["splits"].items()},
              flush=True)
    table: dict[str, Any] = {}
    for chore in CHORES_M7:
        for nm in dict.fromkeys(c.name for c in conds if c.chore == chore):
            rs = [r for r in results if r["chore"] == chore and r["name"] == nm]
            times = [r["time_per_chore_s"] for r in rs if r["time_per_chore_s"]]
            table[f"{chore}/{nm}"] = {
                **{sp: {"success": pooled(rs, sp), "objects": pooled(rs, sp, "objects")} for sp in rs[0]["splits"]},
                "human_min": float(np.mean([r["human"]["total_min"] for r in rs])),
                "time_per_chore_s": float(np.mean(times)) if times else None}
    return {"table": table, "runs": results}


def part_b(seeds: list[int], n: int | None) -> dict:
    chore = "desk_clearing"
    base = load_protocol(chore, "eval_seen")[: n or 20]
    tests: dict[str, Any] = {
        "nominal": ("cards", base, "overhead", None),
        "unseen_objects": ("split", "eval_unseen_obj", "overhead", None),
        "lighting": ("split", "eval_lighting", "overhead", None),
        "clutter_new_colours": ("cards", [add_clutter(c, ["can", "eraser"], c.seed) for c in base], "overhead",
                                ["can", "eraser"]),
        "clutter_same_colours": ("cards", [add_clutter(c, ["block_red", "block_blue"], c.seed) for c in base],
                                 "overhead", ["block_red", "block_blue"]),
        "camera_bump_1cm": ("cards", base, "bump:0.01", None),
        "camera_bump_2cm": ("cards", base, "bump:0.02", None),
    }
    rows: list[dict[str, Any]] = []
    for method in ("video_10", "teleop_1x"):
        for s in seeds:
            pols = load_ckpts(chore, method, s)
            if method == "video_10":
                prog = program_from_records(chore, video_records(chore, training_cards(chore, "film", 10,
                                                                                       offset=100 * s)))
            else:
                prog = teleop_condition(chore, video_condition(chore, 10, s).ledger.total_s(), s, "t").program  # type: ignore[assignment]
            for name, (kind, what, perc, dis) in tests.items():
                cards = what if kind == "cards" else None
                split = what if kind == "split" else "x"
                tr = evaluate(chore, pols, prog, split, n=n if kind == "split" else None, perceiver=perc,
                              hook_kw={"distractors": dis} if dis else None, cards=cards)
                rows.append({"method": method, "seed": s, "test": name, "success": rate(tr), "objects": object_rate(tr),
                             "checkpoint_hash": ensemble_hash(pols),
                             "trials": [{k: t[k] for k in ("seed", "success", "n_done", "n_total")} for t in tr]})
                print("B", method, s, name, rows[-1]["success"]["k"], "/", rows[-1]["success"]["n"], flush=True)
    from eval.stats import wilson

    matrix: dict[str, Any] = {}
    for method in ("video_10", "teleop_1x"):
        for name in tests:
            rs = [r for r in rows if r["method"] == method and r["test"] == name]
            k, nn = sum(r["success"]["k"] for r in rs), sum(r["success"]["n"] for r in rs)
            ko, no = sum(r["objects"]["k"] for r in rs), sum(r["objects"]["n"] for r in rs)
            matrix[f"{method}/{name}"] = {"success": dict(zip(("p", "lo", "hi"), wilson(k, nn))) | {"k": k, "n": nn},
                                          "objects": dict(zip(("p", "lo", "hi"), wilson(ko, no))) | {"k": ko, "n": no}}
    return {"matrix": matrix, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--part", default="ab")
    args = ap.parse_args()
    t0 = time.time()
    seeds = [0] if args.quick else list(range(args.seeds))
    n = 6 if args.quick else None
    towel = ("not run: the digital twin has no deformable-object model; real-robot folding is queued in "
             "SESSION_QUEUE.md (awaiting-human)")
    q = "_quick" if args.quick else ""
    for part, fn in (("a", part_a), ("b", part_b)):   # each part is written as soon as it finishes
        if part in args.part:
            payload: dict[str, Any] = {f"part_{part}": fn(seeds, n), "towel_folding": towel,
                                       "wall_s": time.time() - t0}
            out = write_result(f"sim/m7_multichore{q}_{part}", payload, stage="sim", config=vars(args), seed=0,
                               command=cmdline())
            print("wrote", out)


if __name__ == "__main__":
    main()
