"""M5 (sim stage): uncertainty-driven vs random vs round-robin demo requests at equal demo count / human time.

Per seed: start from the same 10 phone videos; then R rounds, each adding k requested videos chosen by the strategy;
after every round retrain a 3-member bootstrap ensemble (same config and compute for every strategy) and evaluate on
the frozen eval_seen + eval_unseen_place cards. The uncertainty strategy rolls the current ensemble out in the
digital twin on a pool of candidate placements (training-side seeds, never eval) and requests the placement whose
worst object step has the highest ensemble disagreement; random draws from the same pool; round-robin sweeps
objects x mat regions.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from active.requests import (
    candidate_pool,
    pick_random,
    pick_round_robin,
    pick_uncertain,
    render_key_frame,
)
from bench.common import (
    Condition,
    HumanLedger,
    ckey,
    cmdline,
    evaluate,
    evaluate_condition,
    ledger_from_records,
    pooled,
    program_from_records,
    segments_from_records,
    strip_trials,
    train_conditions,
    video_records,
)
from common.provenance import RESULTS_DIR, write_result
from eval.placements import PlacementCard, training_cards
from policies.train import TrainConfig

STRATEGIES = ("uncertainty", "random", "round_robin")
REQUEST_OVERHEAD_S = 10.0   # reading the request + checking the placement card (same for every strategy)


def make_condition(chore: str, strategy: str, seed: int, recs, n_requests: int, members: int) -> Condition:
    prog = program_from_records(chore, recs)
    segs = segments_from_records(recs, "video", prog)
    L = ledger_from_records(recs, ledger=HumanLedger())
    for _ in range(n_requests):
        L.add("request", REQUEST_OVERHEAD_S)
    return Condition(strategy, chore, seed, segs, prog, L, TrainConfig(seed=seed), n_members=members,
                     meta={"n_videos": len(recs), "n_requests": n_requests})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--k", type=int, default=2, help="requests per round")
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.rounds, args.pool, args.members = 1, 1, 8, 2
    n_eval = 10 if args.quick else None
    seeds = list(range(args.seeds))
    t0 = time.time()
    recs = {(st, s): list(video_records(args.chore, training_cards(args.chore, "film", 10, offset=100 * s)))
            for st in STRATEGIES for s in seeds}
    nreq = {(st, s): 0 for st in STRATEGIES for s in seeds}
    rng = {s: np.random.default_rng(1000 + s) for s in seeds}
    curve: list[dict] = []
    requests_log: list[dict] = []
    for r in range(args.rounds + 1):
        # round 0 is identical for all strategies: train it once per seed
        strategies = STRATEGIES if r > 0 else ("uncertainty",)
        conds = [make_condition(args.chore, st, s, recs[(st, s)], nreq[(st, s)], args.members)
                 for st in strategies for s in seeds]
        pols = train_conditions(conds)
        for c in conds:
            res = evaluate_condition(c, pols[ckey(c)], n=n_eval)
            for st in (STRATEGIES if r == 0 else (c.name,)):
                curve.append({**strip_trials(res), "strategy": st, "round": r})
            print(f"round {r} {c.name} seed {c.seed} videos {len(recs[(c.name, c.seed)])} "
                  f"min {res['human']['total_min']}",
                  {sp: (d["success"]["k"], d["success"]["n"], round(d["objects"]["p"], 2))
                   for sp, d in res["splits"].items()}, flush=True)
        if r == args.rounds:
            break
        for s in seeds:
            for st in STRATEGIES:
                base = [c for c in conds if c.seed == s and (c.name == st or r == 0)][0]
                pol = pols[ckey(base)]
                pool = candidate_pool(args.chore, args.pool, r, s)
                if st == "uncertainty":
                    trials = evaluate(args.chore, pol, base.program, "request", cards=pool)
                    reqs = pick_uncertain(pool, trials, args.k, args.chore)
                elif st == "random":
                    reqs = pick_random(pool, args.k, rng[s])
                else:
                    reqs = pick_round_robin(args.chore, nreq[(st, s)], args.k, s)
                for i, q in enumerate(reqs):
                    if s == 0:
                        render_key_frame(q, RESULTS_DIR / "sim" / "m5_requests" / f"{args.chore}_{st}_r{r}_{i}.png")
                    requests_log.append({"seed": s, "round": r, **q.to_json()})
                cards = [PlacementCard.from_json(q.card) for q in reqs]
                new = video_records(args.chore, cards)
                recs[(st, s)] += new
                nreq[(st, s)] += len(new)
    summary: dict = {}
    for st in STRATEGIES:
        summary[st] = []
        for r in range(args.rounds + 1):
            rs = [c for c in curve if c["strategy"] == st and c["round"] == r]
            both: dict[str, dict] = {"success": {}, "objects": {}}
            for sp in rs[0]["splits"]:
                both["success"][sp] = pooled(rs, sp)
                both["objects"][sp] = pooled(rs, sp, "objects")
            k = sum(both["success"][sp]["k"] for sp in rs[0]["splits"])
            n = sum(both["success"][sp]["n"] for sp in rs[0]["splits"])
            from eval.stats import wilson

            p, lo, hi = wilson(k, n)
            summary[st].append({"round": r, "n_videos": 10 + args.k * r,
                                "human_min": float(np.mean([c["human"]["total_min"] for c in rs])),
                                "success_all": {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}, **both})
    out = write_result(f"sim/m5_active_{args.chore}" + ("_quick" if args.quick else ""),
                       {"summary": summary, "curve": curve, "requests": requests_log, "wall_s": time.time() - t0},
                       stage="sim", config=vars(args) | {"train": TrainConfig().__dict__,
                                                         "request_overhead_s": REQUEST_OVERHEAD_S},
                       seed=0, command=cmdline())
    print("wrote", out)


if __name__ == "__main__":
    main()
