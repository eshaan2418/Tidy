"""Queued real-robot sessions (eval / practice), run ONLY via scripts/supervised_session.py.

Session files: data/sessions/<id>.json, written by eval/queue.py: chore, conditions (checkpoint files + inferred
program per method), and a trial list whose order is randomised across methods (cancels drift). Each trial:
  1. operator re-confirms presence (and at least every 5 minutes) and sets up the printed placement card;
  2. the executor runs the chore with the condition's policy (same code as in sim) while the scene camera records;
  3. auto-score from the overhead camera + operator score per object (y/n); disagreements are kept for review.
`--sim` runs the identical protocol in the digital twin (CI dry run; nobody is asked anything).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.provenance import REPO_ROOT, write_result
from eval.chores import CHORES, PlaceGoal
from eval.placements import PlacementCard
from sim.scene import HUES

SESSIONS = REPO_ROOT / "data" / "sessions"
PRESENCE_PERIOD_S = 300.0


def load_session(session: str) -> dict:
    p = SESSIONS / f"{session}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: create it with `make queue-session` (eval/queue.py)")
    return json.loads(p.read_text())


def _policies(cond: dict):
    from policies.train import load_policy

    return [load_policy((REPO_ROOT / c).read_bytes()) for c in cond["checkpoints"]]


def _ask_score(chore, names: list[str]) -> dict[str, bool]:
    out = {}
    for n in names:
        a = input(f"  operator score: is the {n.replace('_', ' ')} at its goal? [y/n] ").strip().lower()
        out[n] = a.startswith("y")
    return out


def run_queued_session(protocol: str, session: str, robot, out_dir: Path, interactive: bool, confirm,
                       recorder_cls) -> dict:
    from bench.common import auto_placed
    from eval.runner import make_env, run_episode
    from perception.overhead import OverheadPerceiver, RayPlanePerceiver
    from phases.program import InferredProgram
    from policies.agent import PolicyAgent

    spec = load_session(session)
    if spec["protocol"] != protocol:
        raise ValueError(f"session {session} is a {spec['protocol']} session, not {protocol}")
    chore = CHORES[spec["chore"]]
    conds = {name: {**c, "pols": _policies(c)} for name, c in spec["conditions"].items()}
    log: dict = {"session": session, "chore": chore.name, "protocol": protocol, "real": interactive, "trials": []}
    last_confirm = time.monotonic()
    videos = []
    for i, tr in enumerate(spec["trials"]):
        cond = conds[tr["condition"]]
        card = PlacementCard.from_json(tr["card"])
        if interactive:
            print(f"\nTrial {i + 1}/{len(spec['trials'])}: set up placement card #{card.seed}: {card.human_text()}")
            confirm(True, "Placement ready, area clear, hand on the e-stop?")
            last_confirm = time.monotonic()
            from hardware.real_env import RealEnv, load_overhead_camera

            perc: Any = RayPlanePerceiver(camera=load_overhead_camera())
            hue_file = REPO_ROOT / "hardware" / "cameras" / "hues.json"
            cal = spec.get("hue_calibration") or (json.loads(hue_file.read_text()) if hue_file.exists() else {})
            perc.hue_cal = {**{k: float(v) for k, v in HUES.items()}, **cal}
            env: Any = RealEnv(robot, chore.object_specs(card.subs), chore.fixture_specs(), perc)
        else:
            perc = OverheadPerceiver()
            env = make_env(chore, card)
        rec = recorder_cls(out_dir, f"trial_{i:03d}_{tr['condition']}")
        orig = env.step

        def step(a, monitor_rgb=None, orig=orig, rec=rec, env=env):
            nonlocal last_confirm
            if interactive and time.monotonic() - last_confirm > PRESENCE_PERIOD_S:
                confirm(True, "Periodic check (5 min): still present and watching?")
                last_confirm = time.monotonic()
            info = orig(a, monitor_rgb)
            rec.add(env.render("overhead", 320, 240))
            return info

        env.step = step  # type: ignore[method-assign]
        prog = cond["program"]
        program = {k: PlaceGoal(v[0], tuple(v[1])) for k, v in prog["program"].items()}
        agent = PolicyAgent(cond["pols"], seed=i)
        res = run_episode(chore, card, agent, perc, program=program, order=prog.get("order"), env=env,
                          record=protocol == "practice")
        if protocol == "practice":   # keep every practice step (auto-score attached) for retraining
            pdir = REPO_ROOT / "data" / "robot" / f"practice_{session}"
            pdir.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, Any] = {f"{k}_{s.obj}_{j}": np.array(getattr(s, k)) for j, s in enumerate(res.steps)
                                      for k in ("obs", "act")}
            np.savez_compressed(pdir / f"trial_{i:03d}.npz", **arrays)
        video = rec.close()
        videos.append(video)
        ip = InferredProgram(program, prog.get("order"), [], {}, 0)
        auto = {s.obj: auto_placed(chore, s, ip) for s in res.steps}
        human = _ask_score(chore, chore.movable()) if interactive else dict(res.per_object)
        log["trials"].append({"i": i, "condition": tr["condition"], "card_seed": card.seed, "split": card.split,
                              "auto": auto, "operator": human, "success": all(human.values()),
                              "agree": {n: auto.get(n) == human.get(n) for n in human},
                              "safety_events": res.safety_events, "ticks": res.ticks, "video": video})
        env.close()
    from eval.stats import wilson

    summary = {}
    for name in conds:
        ts = [t for t in log["trials"] if t["condition"] == name]
        k = sum(t["success"] for t in ts)
        summary[name] = dict(zip(("p", "lo", "hi"), wilson(k, len(ts)))) | {"k": k, "n": len(ts)}
    agree = [v for t in log["trials"] for v in t["agree"].values()]
    log["summary"] = summary
    log["auto_vs_operator_agreement"] = float(np.mean(agree)) if agree else None
    log["pass"] = True
    write_result(("real/" if interactive else "sim/session_dryrun_") + session, log,
                 stage="real" if interactive else "sim", config={"session_file": spec}, seed=None,
                 videos=videos, command=f"scripts/supervised_session.py {protocol} --session {session}")
    return log
