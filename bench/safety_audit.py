"""Safety audit (sim): replay trained policies and check the MEASURED arm state, not just the commands.

Per control tick: measured TCP inside the workspace box (tolerance 3 mm for servo overshoot) and outside every
no-go zone; measured joint positions inside the shrunk joint limits; measured joint speed (|dq|/dt between ticks)
<= the per-joint limit x 1.1; TCP speed <= the Cartesian limit x 1.1. Also counts safety-filter interventions
(commands the filter clamped or refused). An "incident" is any measured-state violation.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from bench.common import _WORKER_STATE, WORKERS, _init_worker, cmdline
from bench.m6_recovery import load_ckpts
from common.provenance import write_result
from eval.chores import CHORES
from eval.placements import PlacementCard, load_protocol, training_cards
from eval.runner import make_env, run_episode
from perception.overhead import OverheadPerceiver
from safety.limits import DEFAULT_SAFETY, in_no_go
from sim.env import CONTROL_DT

TOL = 0.003


def _audit_episode(args) -> dict:
    chore_name, card_json, program, order, seed = args
    from policies.agent import PolicyAgent

    chore = CHORES[chore_name]
    card = PlacementCard.from_json(card_json)
    env = make_env(chore, card)
    agent = PolicyAgent(_WORKER_STATE["pols"], seed=seed)
    cfg = DEFAULT_SAFETY
    stats = {"ticks": 0, "workspace": 0, "no_go": 0, "joint_limit": 0, "joint_speed": 0, "tcp_speed": 0,
             "interventions": 0, "max_joint_speed_frac": 0.0, "max_tcp_speed": 0.0}
    orig = env.step
    prev: dict[str, Any] = {"q": None, "p": None}

    def step(a, monitor_rgb=None):
        n_before = len(env.safety_events)
        info = orig(a, monitor_rgb)
        stats["interventions"] += int(len(env.safety_events) > n_before)
        q = env.joint_pos()
        p, _, _ = env.ee_state()
        stats["ticks"] += 1
        lo, hi = np.array(cfg.workspace.lo), np.array(cfg.workspace.hi)
        stats["workspace"] += int(np.any(p < lo - TOL) or np.any(p > hi + TOL))
        stats["no_go"] += int(in_no_go(p, cfg))
        stats["joint_limit"] += int(np.any(q < cfg.q_lo - 0.02) or np.any(q > cfg.q_hi + 0.02))
        if prev["q"] is not None:
            js = np.abs(q - prev["q"]) / CONTROL_DT / np.array(cfg.max_joint_vel)
            ts = float(np.linalg.norm(p - prev["p"]) / CONTROL_DT)
            stats["max_joint_speed_frac"] = max(stats["max_joint_speed_frac"], float(js.max()))
            stats["max_tcp_speed"] = max(stats["max_tcp_speed"], ts)
            stats["joint_speed"] += int(js.max() > 1.1)
            stats["tcp_speed"] += int(ts > 1.1 * cfg.max_tcp_speed)
        prev["q"], prev["p"] = q, p
        return info

    env.step = step  # type: ignore[method-assign]
    env.reset(card.placement)
    prev["q"], prev["p"] = env.joint_pos(), env.ee_state()[0]
    run_episode(chore, card, agent, OverheadPerceiver(), program=program, order=order, env=env)
    env.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chore", default="desk_clearing")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    from bench.common import program_from_records, video_records

    n = 3 if a.quick else a.n
    rows = []
    for method in ("video_10", "teleop_1x", "video_raw"):
        for s in ([0] if a.quick else [0, 1, 2]):
            try:
                pols = load_ckpts(a.chore, method, s)
            except FileNotFoundError:
                continue
            prog = program_from_records(a.chore, video_records(a.chore, training_cards(a.chore, "film", 10,
                                                                                       offset=100 * s)))
            cards = load_protocol(a.chore, "eval_seen")[:n] + load_protocol(a.chore, "eval_unseen_place")[:n]
            jobs = [(a.chore, c.to_json(), prog.program, prog.order, 7 + i) for i, c in enumerate(cards)]
            with ProcessPoolExecutor(WORKERS, initializer=_init_worker,
                                     initargs=([p.checkpoint_bytes() for p in pols], {})) as ex:
                eps = list(ex.map(_audit_episode, jobs))
            tot = {k: (max(e[k] for e in eps) if k.startswith("max") else sum(e[k] for e in eps)) for k in eps[0]}
            rows.append({"method": method, "seed": s, "episodes": len(eps), **tot})
            print(rows[-1], flush=True)
    keys = ("workspace", "no_go", "joint_limit", "joint_speed", "tcp_speed")
    total = {"episodes": sum(r["episodes"] for r in rows), "ticks": sum(r["ticks"] for r in rows),
             "incidents": sum(r[k] for r in rows for k in keys),
             "interventions": sum(r["interventions"] for r in rows),
             "max_joint_speed_frac": max(r["max_joint_speed_frac"] for r in rows),
             "max_tcp_speed": max(r["max_tcp_speed"] for r in rows)}
    write_result("sim/safety_audit" + ("_quick" if a.quick else ""), {"rows": rows, "total": total,
                                                                      "wall_s": time.time() - t0},
                 stage="sim", config={"tolerance_m": TOL, "speed_factor": 1.1, **vars(a)}, seed=0, command=cmdline())
    print(total)


if __name__ == "__main__":
    main()
