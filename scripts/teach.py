"""Tidy teaching app (M8, minimal CLI). Teach the robot a new chore from phone videos.

    python scripts/teach.py new tray_heldout            # 1. new chore
    python scripts/teach.py guide tray_heldout          # 2. filming guide + placement cards to film
    python scripts/teach.py upload tray_heldout DIR     # 3. copy your clips (real) | --sim N: simulated person
    python scripts/teach.py process tray_heldout        # 4. processing progress + quality report + "what I learned"
    python scripts/teach.py train tray_heldout          # 5. train ("Ready to try")
    python scripts/teach.py try tray_heldout            # 6. supervised run: real = queues a session for
                                                        #    scripts/supervised_session.py; --sim = digital twin
    python scripts/teach.py requests tray_heldout       # 7. the robot's demo requests (text + key frame)
    python scripts/teach.py record tray_heldout DIR     # 8. add the requested demo(s) (real) | --sim
    python scripts/teach.py status tray_heldout         # human minutes so far (every minute counts, Rule 5)
    python scripts/teach.py walkthrough tray_heldout --sim   # run 1-8 end to end in the twin (scripted user)

State lives in data/human/<chore>/teach_state.json (no video). Raw clips stay in data/private/videos/<chore>/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.provenance import write_result  # noqa: E402

APP_STEP_MIN = 0.5   # assumed human minutes per app screen (reading/clicking), logged for every step


def state_path(chore: str) -> Path:
    return ROOT / "data" / "human" / chore / "teach_state.json"


def load(chore: str) -> dict:
    p = state_path(chore)
    if not p.exists():
        raise SystemExit(f"unknown chore '{chore}': run `teach.py new {chore}` first")
    return json.loads(p.read_text())


def save(st: dict) -> None:
    p = state_path(st["chore"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1))


def log_minutes(st: dict, step: str, minutes: float) -> None:
    st["human_log"].append({"step": step, "minutes": round(float(minutes), 3), "t": time.time()})


def cmd_new(a) -> dict:
    from eval.chores import CHORES

    if a.chore not in CHORES:
        raise SystemExit(f"'{a.chore}' must match an object/fixture set in eval/chores.py (the perception library); "
                         "add it there first (objects + fixtures only; the goal is learned from your videos)")
    st = {"chore": a.chore, "objects": CHORES[a.chore].objects, "fixtures": CHORES[a.chore].fixtures, "film_cards": [],
          "videos": [], "requests": [], "checkpoints": [], "human_log": [], "status": "new"}
    log_minutes(st, "new", APP_STEP_MIN)
    save(st)
    print(f"New chore '{a.chore}': objects {st['objects']}, fixtures {st['fixtures']}. Next: teach.py guide")
    return st


def cmd_guide(a) -> dict:
    from eval.placements import training_cards

    st = load(a.chore)
    cards = training_cards(a.chore, "film", a.n)
    st["film_cards"] = [c.to_json() for c in cards]
    print("FILMING GUIDE (full version: HUMAN_TODO.md H5)\n"
          " - phone landscape, from your side of the table, ~60 cm away and ~45 cm up, all 4 mat corners visible\n"
          " - robot powered off and parked; do the chore once per clip at a natural pace; hands only, no tools\n"
          " - between clips, re-arrange the items exactly as on the next placement card")
    for i, c in enumerate(cards):
        print(f"  clip {i + 1:02d}: {c.human_text()}")
    log_minutes(st, "guide", APP_STEP_MIN)
    st["status"] = "filming"
    save(st)
    return st


def _sim_videos(st: dict, cards_json: list[dict]) -> list[dict]:
    from bench.common import video_records
    from eval.placements import PlacementCard

    recs = video_records(st["chore"], [PlacementCard.from_json(c) for c in cards_json])
    out = []
    for c, r in zip(cards_json, recs):
        out.append({"card": c, "source": "sim", "accepted": r.accepted, "reasons": r.reject_reasons,
                    "film_min": r.video_duration_s / 60, "reset_min": 20 / 60, "setup_min": 6 / 60})
    return out


def cmd_upload(a) -> dict:
    st = load(a.chore)
    if a.sim:
        new = _sim_videos(st, st["film_cards"][: a.n])
    else:
        dst = ROOT / "data" / "private" / "videos" / a.chore
        dst.mkdir(parents=True, exist_ok=True)
        new = []
        for i, clip in enumerate(sorted(Path(a.dir).iterdir())):
            shutil.copytree(clip, dst / clip.name, dirs_exist_ok=True) if clip.is_dir() else shutil.copy(clip, dst)
            new.append({"card": st["film_cards"][i] if i < len(st["film_cards"]) else None, "source": str(clip.name)})
        minutes = float(input("How many minutes did filming take, including re-arranging items? "))
        log_minutes(st, "film", minutes)
    for v in new:
        if a.sim:
            log_minutes(st, "film", v["film_min"] + v["setup_min"])
            log_minutes(st, "reset", v["reset_min"])
    st["videos"] += new
    log_minutes(st, "upload", APP_STEP_MIN)
    st["status"] = "uploaded"
    save(st)
    print(f"{len(new)} clips added ({len(st['videos'])} total). Next: teach.py process")
    return st


def _records(st: dict):
    from bench.common import video_records
    from eval.placements import PlacementCard

    cards = [PlacementCard.from_json(v["card"]) for v in st["videos"] if v.get("source") == "sim"]
    if len(cards) != len(st["videos"]):
        raise SystemExit("real clips: run perception.real_video.process_clip on data/private/videos/<chore>/ (needs "
                         "fingertip caps or a hand-pose model; first real run is HUMAN_TODO.md H5/H6)")
    return video_records(st["chore"], cards)


def cmd_process(a) -> dict:
    from bench.common import program_from_records

    st = load(a.chore)
    recs = _records(st)
    for i, r in enumerate(recs):
        print(f"  [{i + 1}/{len(recs)}] {'ok      ' if r.accepted else 'REJECTED'} "
              f"{len(r.segments)} pick-place(s) {'; '.join(r.reject_reasons)}")
    prog = program_from_records(st["chore"], recs)
    print("What I learned from your videos:")
    for o, g in prog.program.items():
        print(f"  - {o.replace('_', ' ')} -> {g.landmark.split(':')[1].replace('_', ' ')} "
              f"(spread {100 * prog.spread[o]:.1f} cm)")
    for o in prog.static:
        print(f"  - {o.replace('_', ' ')} stays where it is")
    if prog.order:
        print(f"  - order: {' then '.join(prog.order)}")
    st["program"] = {k: [v.landmark, list(v.offset)] for k, v in prog.program.items()}
    st["order"] = prog.order
    log_minutes(st, "process", APP_STEP_MIN)
    st["status"] = "processed"
    save(st)
    return st


def cmd_train(a) -> dict:
    from bench.common import Condition, HumanLedger, ckey, program_from_records, segments_from_records, train_conditions
    from policies.train import TrainConfig

    st = load(a.chore)
    recs = _records(st)
    prog = program_from_records(st["chore"], recs)
    segs = segments_from_records(recs, "video", prog)
    c = Condition(f"teach_v{len(recs)}", st["chore"], 0, segs, prog, HumanLedger(), TrainConfig(seed=0), n_members=3)
    pols = train_conditions([c])[ckey(c)]
    st["checkpoints"] = [f"checkpoints/{st['chore']}__{c.name}__s0__m{i}.pt" for i in range(len(pols))]
    log_minutes(st, "train", APP_STEP_MIN)
    st["status"] = "ready_to_try"
    save(st)
    print(f"Trained a 3-member ensemble on {len(segs)} pseudo-demo segments. Ready to try: teach.py try")
    return st


def _load_policies(st: dict):
    from policies.train import load_policy

    return [load_policy((ROOT / p).read_bytes()) for p in st["checkpoints"]]


def _program(st: dict):
    from eval.chores import PlaceGoal
    from phases.program import InferredProgram

    prog = {k: PlaceGoal(v[0], tuple(v[1])) for k, v in st["program"].items()}
    return InferredProgram(prog, st.get("order"), [], {}, 0)


def cmd_try(a) -> dict:
    from bench.common import evaluate, object_rate, rate

    st = load(a.chore)
    if not a.sim:
        print("Real robot: queue a supervised session and run it yourself, present at the robot:\n"
              f"  python -m eval.queue --chore {a.chore} --session T_{a.chore} "
              f"--conditions teach_v{len(st['videos'])}\n"
              f"  python scripts/supervised_session.py eval --session T_{a.chore}")
        return st
    trials = evaluate(st["chore"], _load_policies(st), _program(st), "eval_seen", n=a.n)
    r, o = rate(trials), object_rate(trials)
    print(f"Twin try-out on {a.n} unseen placements: {r['k']}/{r['n']} full chores, {100 * o['p']:.0f}% of items")
    st.setdefault("tries", []).append({"n_videos": len(st["videos"]), "success": r, "objects": o})
    log_minutes(st, "try", APP_STEP_MIN + (sum(t["ticks"] for t in trials) / 600 if a.sim else 0))  # supervision
    save(st)
    return st


def cmd_requests(a) -> dict:
    from active.requests import candidate_pool, pick_uncertain, render_key_frame
    from bench.common import evaluate

    st = load(a.chore)
    pool = candidate_pool(st["chore"], a.pool, len(st["requests"]), 0)
    trials = evaluate(st["chore"], _load_policies(st), _program(st), "request", cards=pool)
    reqs = pick_uncertain(pool, trials, a.k, st["chore"])
    for i, q in enumerate(reqs):
        kf = render_key_frame(q, ROOT / "data" / "human" / st["chore"] / f"request_{len(st['requests']) + i}.png")
        print(f"  Request {i + 1}: {q.text}\n    key frame: {kf}")
        st["requests"].append(q.to_json())
    log_minutes(st, "requests", APP_STEP_MIN)
    save(st)
    return st


def cmd_record(a) -> dict:
    st = load(a.chore)
    pending = [q for q in st["requests"] if not q.get("recorded")]
    if a.sim:
        new = _sim_videos(st, [q["card"] for q in pending])
        for v in new:
            log_minutes(st, "film_request", v["film_min"] + v["setup_min"] + 10 / 60)
            log_minutes(st, "reset", v["reset_min"])
        st["videos"] += new
    else:
        print("Copy the requested clip(s) into the folder you pass, then re-run process/train.")
        return cmd_upload(a)
    for q in pending:
        q["recorded"] = True
    save(st)
    print(f"Recorded {len(pending)} requested demo(s). Next: teach.py process, train, try")
    return st


def cmd_status(a) -> dict:
    st = load(a.chore)
    by: dict[str, float] = {}
    for e in st["human_log"]:
        by[e["step"]] = by.get(e["step"], 0.0) + e["minutes"]
    print(f"{st['chore']}: status {st['status']}, {len(st['videos'])} videos, "
          f"{sum(by.values()):.1f} human minutes: " + ", ".join(f"{k} {v:.1f}" for k, v in by.items()))
    return st


def cmd_walkthrough(a) -> dict:
    """Scripted non-expert session in the twin: exactly the commands a user would type, in order."""
    if not a.sim:
        raise SystemExit("walkthrough is the scripted sim run; with the real robot, follow the steps one by one")
    t0 = time.time()
    ns = argparse.Namespace
    cmd_new(ns(chore=a.chore))
    cmd_guide(ns(chore=a.chore, n=10))
    cmd_upload(ns(chore=a.chore, sim=True, n=10, dir=None))
    cmd_process(ns(chore=a.chore))
    cmd_train(ns(chore=a.chore))
    cmd_try(ns(chore=a.chore, sim=True, n=a.n))
    cmd_requests(ns(chore=a.chore, pool=20, k=2))
    cmd_record(ns(chore=a.chore, sim=True))
    cmd_process(ns(chore=a.chore))
    cmd_train(ns(chore=a.chore))
    st = cmd_try(ns(chore=a.chore, sim=True, n=a.n))
    cmd_status(ns(chore=a.chore))
    by: dict[str, float] = {}
    for e in st["human_log"]:
        by[e["step"]] = by.get(e["step"], 0.0) + e["minutes"]
    write_result(f"sim/m8_teach_{a.chore}", {"state": st, "human_minutes_total": sum(by.values()),
                                              "human_minutes_by_step": by, "wall_s": time.time() - t0},
                 stage="sim", config={"app_step_min": APP_STEP_MIN, "n_try": a.n}, seed=0,
                 command=" ".join(["python", "scripts/teach.py"] + sys.argv[1:]))
    return st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("new", "guide", "upload", "process", "train", "try", "requests", "record", "status", "walkthrough"):
        p = sub.add_parser(name)
        p.add_argument("chore")
        p.add_argument("dir", nargs="?")
        p.add_argument("--sim", action="store_true")
        p.add_argument("--n", type=int, default=10 if name in ("guide", "upload") else 20)
        p.add_argument("--k", type=int, default=2)
        p.add_argument("--pool", type=int, default=20)
    a = ap.parse_args(argv)
    {"new": cmd_new, "guide": cmd_guide, "upload": cmd_upload, "process": cmd_process, "train": cmd_train,
     "try": cmd_try, "requests": cmd_requests, "record": cmd_record, "status": cmd_status,
     "walkthrough": cmd_walkthrough}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
