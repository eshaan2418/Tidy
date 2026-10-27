"""M6 (sim stage): per-phase failure detection + reset-and-retry recovery.

1. Detector data: roll out trained policies (video_5 / video_10 / teleop_1x checkpoints from M4, for a mix of
   failure modes) on `practice` cards (training-side seeds). Every object step gives robot-observable features
   (recovery.recover.step_features) and simulator labels (grasped / placed).
2. Fit grasp- and place-phase logistic detectors on 70 % of the steps, calibrate the threshold on the other 30 %
   (target precision 0.8), and measure precision / recall on steps from separate eval-card rollouts.
3. Full-chore success with vs without recovery (max 2 retries per object, then "ask for help"), plus an oracle
   detector (simulator truth) as the upper bound and a detector-free "retry every step once" baseline (is the
   detector needed, or do extra attempts alone help?).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from bench.common import (
    _WORKER_STATE,
    WORKERS,
    _init_worker,
    cmdline,
    evaluate,
    object_rate,
    program_from_records,
    rate,
    video_records,
)
from common.provenance import REPO_ROOT, write_result
from eval.chores import CHORES
from eval.placements import PlacementCard, load_protocol, training_cards
from eval.runner import make_env, run_episode
from perception.overhead import OverheadPerceiver
from policies.train import CKPT_DIR, ensemble_hash, load_policy
from recovery.recover import FEATURES, FailureDetector, step_features

DET_PATH = REPO_ROOT / "results" / "sim" / "m6_detector_{chore}.json"


def _steps_episode(args) -> list[dict]:
    chore_name, card_json, program, order, seed = args
    from policies.agent import PolicyAgent

    chore = CHORES[chore_name]
    card = PlacementCard.from_json(card_json)
    env = make_env(chore, card)
    agent = PolicyAgent(_WORKER_STATE["pols"], seed=seed)
    res = run_episode(chore, card, agent, OverheadPerceiver(), program=program, order=order, env=env)
    out = [{"x": step_features(s, env.obj_by_name[s.obj]).tolist(), "grasped": bool(s.grasped),
            "placed": bool(s.placed), "obj": s.obj} for s in res.steps]
    env.close()
    return out


def collect_steps(chore: str, pols, prog, cards: list[PlacementCard]) -> list[dict]:
    jobs = [(chore, c.to_json(), prog.program, prog.order, 51 + i) for i, c in enumerate(cards)]
    blobs = [p.checkpoint_bytes() for p in pols]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(blobs, {})) as ex:
        return [s for ep in ex.map(_steps_episode, jobs) for s in ep]


def pr(det: FailureDetector, steps: list[dict]) -> dict:
    X = np.array([s["x"] for s in steps])
    fail = np.array([not s["placed"] for s in steps])
    flags = np.array([det.judge(x[None])[0] for x in X])
    phase_pred = np.array([det.judge(x[None])[1] for x in X])
    true_phase = np.array(["grasp" if not s["grasped"] else "place" for s in steps])
    tp = int((flags & fail).sum())
    prec = tp / max(1, int(flags.sum()))
    rec = tp / max(1, int(fail.sum()))
    phase_acc = float(np.mean(phase_pred[fail & flags] == true_phase[fail & flags])) if (fail & flags).any() else None
    return {"n_steps": len(steps), "n_fail": int(fail.sum()), "flagged": int(flags.sum()), "precision": prec,
            "recall": rec, "false_alarm_rate": float((flags & ~fail).sum() / max(1, int((~fail).sum()))),
            "phase_accuracy_on_true_positives": phase_acc}


def load_ckpts(chore: str, name: str, seed: int) -> list:
    paths = sorted(CKPT_DIR.glob(f"{chore}__{name}__s{seed}__m*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints for {chore}/{name}/s{seed}: run bench.m4_video_vs_teleop first")
    return [load_policy(p.read_bytes()) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    seeds = [0] if args.quick else list(range(args.seeds))
    n_prac = 10 if args.quick else 40
    progs = {s: program_from_records(args.chore, video_records(args.chore, training_cards(args.chore, "film", 10,
                                                                                         offset=100 * s)))
             for s in seeds}
    # ---- detector training data (practice cards, several policies for diverse failures)
    train_steps: list[dict] = []
    sources = ["video_10"] if args.quick else ["video_5", "video_10", "teleop_1x"]
    for i, name in enumerate(sources):
        pols = load_ckpts(args.chore, name, 0)
        train_steps += collect_steps(args.chore, pols, progs[0],
                                     training_cards(args.chore, "practice", n_prac, offset=5000 + 1000 * i))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(train_steps))
    cut = int(0.7 * len(idx))
    tr = [train_steps[i] for i in idx[:cut]]
    va = [train_steps[i] for i in idx[cut:]]
    X = np.array([s["x"] for s in tr])
    yg = np.array([s["grasped"] for s in tr], float)
    yp = np.array([s["placed"] for s in tr], float)
    det = FailureDetector.fit(X, yg, yp, np.array([s["x"] for s in va]), np.array([s["placed"] for s in va], float))
    det.meta = {"features": FEATURES, "n_train": len(tr), "n_val": len(va), "sources": sources}
    det_path = Path(str(DET_PATH).format(chore=args.chore))
    det_path.parent.mkdir(parents=True, exist_ok=True)
    det.save(det_path)
    # ---- detector test: eval-card rollouts of each seed's video_10 policy (no recovery)
    test_steps: list[dict] = []
    runs: list[dict[str, Any]] = []
    n_eval = 10 if args.quick else None
    for s in seeds:
        pols = load_ckpts(args.chore, "video_10", s)
        cards = load_protocol(args.chore, "eval_seen") + load_protocol(args.chore, "eval_unseen_place")
        if n_eval:
            cards = cards[:n_eval]
        test_steps += collect_steps(args.chore, pols, progs[s], cards)
        row: dict[str, Any] = {"seed": s, "checkpoint_hash": ensemble_hash(pols)}
        for mode, hook, kw in (("no_recovery", None, None), ("recovery", "recovery", {"detector_path": str(det_path)}),
                               ("oracle_recovery", "recovery", {"detector_path": str(det_path), "oracle": True}),
                               ("blind_retry_once", "recovery", {"detector_path": str(det_path), "blind": True})):
            trials = evaluate(args.chore, pols, progs[s], "x", perceiver="overhead", hook=hook, hook_kw=kw,
                              cards=cards)
            retries = [sum(1 for e in t.get("recovery", []) if e["action"] == "retry") for t in trials]
            helps = [sum(1 for e in t.get("recovery", []) if e["action"] == "help") for t in trials]
            row[mode] = {"success": rate(trials), "objects": object_rate(trials),
                         "mean_retries": float(np.mean(retries)), "mean_help_requests": float(np.mean(helps)),
                         "mean_ticks": float(np.mean([t["ticks"] for t in trials])),
                         "safety_events": int(sum(len(t["safety_events"]) for t in trials)),
                         "trials": [{k: t[k] for k in ("seed", "split", "success", "n_done", "ticks")} for t in trials]}
            print(s, mode, row[mode]["success"]["k"], "/", row[mode]["success"]["n"], "retries",
                  row[mode]["mean_retries"], flush=True)
        runs.append(row)
    from eval.stats import wilson

    pooled: dict[str, Any] = {}
    for mode in ("no_recovery", "recovery", "oracle_recovery", "blind_retry_once"):
        k = sum(r[mode]["success"]["k"] for r in runs)
        n = sum(r[mode]["success"]["n"] for r in runs)
        p, lo, hi = wilson(k, n)
        pooled[mode] = {"k": k, "n": n, "p": p, "lo": lo, "hi": hi,
                        "mean_retries": float(np.mean([r[mode]["mean_retries"] for r in runs])),
                        "safety_events": int(sum(r[mode]["safety_events"] for r in runs))}
    payload: dict[str, Any] = {"detector": {"threshold": det.threshold, "val": pr(det, va), "test": pr(det, test_steps),
                            "train_meta": det.meta},
               "pooled": pooled, "runs": runs, "wall_s": time.time() - t0}
    out = write_result(f"sim/m6_recovery_{args.chore}" + ("_quick" if args.quick else ""), payload, stage="sim",
                       config=vars(args), seed=0, command=cmdline())
    print("detector", payload["detector"]["test"], "\npooled", pooled, "\nwrote", out)


if __name__ == "__main__":
    main()
