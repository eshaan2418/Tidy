"""Queue a real-robot evaluation session: writes data/sessions/<id>.json and appends a row to SESSION_QUEUE.md.

    python -m eval.queue --chore desk_clearing --session S3 --conditions video_10 teleop_1x --n-seen 20 --n-unseen 20

Conditions refer to checkpoint name stems in checkpoints/ (`<chore>__<name>__s<seed>__m*.pt`), e.g. the policies
trained from the REAL phone videos / REAL teleop demos once they exist (HUMAN_TODO H5/H7). Trial order is
randomised across conditions with a fixed seed; every condition gets the same placement cards.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common.provenance import REPO_ROOT
from eval.placements import load_protocol
from eval.session_protocols import SESSIONS
from policies.train import CKPT_DIR


def build_session(chore: str, session: str, conditions: list[str], n_seen: int, n_unseen: int, seed: int,
                  programs: dict[str, dict], protocol: str = "eval") -> dict:
    cards = load_protocol(chore, "eval_seen")[:n_seen] + load_protocol(chore, "eval_unseen_place")[:n_unseen]
    conds = {}
    for c in conditions:
        ck = sorted(CKPT_DIR.glob(f"{chore}__{c}__s{seed}__m*.pt"))
        conds[c] = {"checkpoints": [str(p.relative_to(REPO_ROOT)) for p in ck], "program": programs[c]}
    trials = [{"condition": c, "card": card.to_json()} for card in cards for c in conditions]
    rng = np.random.default_rng(12345)
    order = rng.permutation(len(trials))
    return {"session": session, "chore": chore, "protocol": protocol, "conditions": conds,
            "trials": [trials[i] for i in order], "hue_calibration": None,
            "expected_minutes": round(len(trials) * 1.5, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--session", default="S3")
    ap.add_argument("--conditions", nargs="+", default=["video_10", "teleop_1x"])
    ap.add_argument("--n-seen", type=int, default=20)
    ap.add_argument("--n-unseen", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from bench.common import program_from_records, teleop_data, video_records
    from eval.placements import training_cards

    programs = {}
    for c in args.conditions:
        if c.startswith("teleop"):
            budget = sum(r.human_time_s for r in video_records(args.chore, training_cards(args.chore, "film", 10)))
            prog = teleop_data(args.chore, budget, seed=args.seed)[3]
        else:
            prog = program_from_records(args.chore, video_records(args.chore, training_cards(args.chore, "film", 10)))
        programs[c] = {"program": {k: [v.landmark, list(v.offset)] for k, v in prog.program.items()},
                       "order": prog.order}
    spec = build_session(args.chore, args.session, args.conditions, args.n_seen, args.n_unseen, args.seed, programs)
    SESSIONS.mkdir(parents=True, exist_ok=True)
    (SESSIONS / f"{args.session}.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote data/sessions/{args.session}.json: {len(spec['trials'])} trials, ~{spec['expected_minutes']} min")


if __name__ == "__main__":
    main()
