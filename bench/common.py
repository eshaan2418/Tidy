"""Shared experiment plumbing: data builders (video / teleop / sim pre-training / practice), parallel evaluation,
human-time accounting.  Every bench/*.py experiment composes these and writes results via common.provenance.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from eval.chores import CHORES, ChoreSpec, PlaceGoal, object_at_goal
from eval.placements import PlacementCard, load_protocol, sample_card, training_cards
from eval.runner import EpisodeResult, GTNoisyPerceiver, make_env, run_episode, run_object_step
from perception.overhead import OverheadPerceiver
from phases.program import InferredProgram, ObservedDemo, infer_program
from policies.data import EE_POS, GOAL_REL, Segment
from sim.scene import FIXTURES

RESET_S = 20.0          # human re-placing objects from a placement card (same for every method; assumption)
SETUP_S = 6.0           # start/stop recording (phone) or engage/disengage the leader arm (teleop)
WORKERS = max(1, min(16, (os.cpu_count() or 4) - 2))


# ------------------------------------------------------------------------------------------ human time
@dataclass
class HumanLedger:
    """Every human second spent teaching (Rule 5). Categories: film, teleop, reset, request, practice_reset."""
    entries: list[tuple[str, float]] = field(default_factory=list)

    def add(self, kind: str, seconds: float) -> None:
        self.entries.append((kind, float(seconds)))

    def total_s(self) -> float:
        return float(sum(s for _, s in self.entries))

    def by_kind(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, s in self.entries:
            out[k] = out.get(k, 0.0) + s
        return {k: round(v / 60, 2) for k, v in out.items()}

    def summary(self) -> dict:
        return {"total_min": round(self.total_s() / 60, 2), "by_kind_min": self.by_kind()}


# ------------------------------------------------------------------------------------------ video data
def _video_jobs(chore: str, cards: list[PlacementCard], variant: str, noise=None, rgb_only: bool = False,
                hand_source: str = "sim_estimator") -> list[tuple]:
    return [(chore, c, 7 + c.seed % 100_000, variant, hand_source, noise, rgb_only, True) for c in cards]


def video_records(chore: str, cards: list[PlacementCard], variant: str = "full", noise=None, rgb_only: bool = False,
                  hand_source: str = "sim_estimator"):
    from perception.video_dataset import process_many

    return process_many(_video_jobs(chore, cards, variant, noise, rgb_only, hand_source), workers=WORKERS)


def demo_goal(goal: PlaceGoal, observed: ObservedDemo, k: int) -> np.ndarray | None:
    """The place target the runtime executor would compute for pick k of an observed demo (same definition as
    eval.runner.run_object_step: fixture surface + offset, or landmark object's base + offset)."""
    kind, name = goal.landmark.split(":", 1)
    if kind == "fixture":
        base = observed.fixtures.get(name)
    else:
        base = observed.object_positions_at_place[k].get(name)
    return None if base is None else np.asarray(base, float) + np.array(goal.offset)


def segments_from_records(recs, source: str = "video", program: InferredProgram | None = None) -> list[Segment]:
    """Pseudo-demo segments of accepted clips. With `program`, the goal features are relabelled to the inferred
    task program's goal (D-022), so training goals have the same definition as runtime goals (e.g., a bin goal is
    its rim, where the object is released, not the floor where the object came to rest)."""
    segs = []
    for r in recs:
        if not r.accepted:
            continue
        ks = [k for k, rs in enumerate(r.segments) if rs.accepted]
        for k, (obj, obs, act) in zip(ks, r.pairs):
            if len(obs) <= 3:
                continue
            o = obs
            if program is not None and r.observed is not None:
                if obj not in program.program:
                    continue
                g = demo_goal(program.program[obj], r.observed, k)
                if g is not None:
                    o = obs.copy()
                    o[:, GOAL_REL] = (g[None] - o[:, EE_POS] / 5.0) * 10.0
            segs.append(Segment(o, act, source, obj, {"card": r.card["seed"]}))
    return segs


def ledger_from_records(recs, kind: str = "film", ledger: HumanLedger | None = None) -> HumanLedger:
    L = ledger or HumanLedger()
    for r in recs:  # rejected clips still cost the human their time
        L.add(kind, r.video_duration_s + SETUP_S)
        L.add("reset", RESET_S)
    return L


def program_from_records(chore: str, recs) -> InferredProgram:
    return infer_program([r.observed for r in recs if r.accepted and r.observed is not None], CHORES[chore].objects)


def video_segments(chore: str, n: int, variant: str = "full", offset: int = 0, noise=None, rgb_only: bool = False,
                   hand_source: str = "sim_estimator"):
    cards = training_cards(chore, "film", n, offset=offset)
    recs = video_records(chore, cards, variant, noise, rgb_only, hand_source)
    prog = program_from_records(chore, recs)
    return segments_from_records(recs, "video" if variant == "full" else f"video_{variant}", prog), recs


# ------------------------------------------------------------------------------------------ teleop data
def _teleop_episode(args):
    chore_name, card_json, seed = args
    from sim.teleop import TeleopAgent

    chore = CHORES[chore_name]
    card = PlacementCard.from_json(card_json)
    env = make_env(chore, card)
    agent = TeleopAgent(env, seed=seed)   # the operator acts on the true scene (+ judgement noise)
    # recorded observations come from the robot's own camera, exactly as they would on the real robot
    res = run_episode(chore, card, agent, OverheadPerceiver(), env=env, record=True)
    env.close()
    return res, card_json


def teleop_data(chore: str, budget_s: float, seed: int = 0, max_demos: int = 40):
    """Simulated leader-arm demos on `teleop` cards until the human-time budget is used up (Rule 6).
    Returns (segments of successfully placed object steps, ledger, episodes, inferred program)."""
    cards = training_cards(chore, "teleop", max_demos, offset=1000 * seed)
    jobs = [(chore, c.to_json(), seed * 1000 + i) for i, c in enumerate(cards)]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(_teleop_episode, jobs))
    L = HumanLedger()
    segs: list[Segment] = []
    episodes: list[EpisodeResult] = []
    observed: list[ObservedDemo] = []
    for res, cj in outs:
        cost = res.ticks * 0.1 + SETUP_S + RESET_S
        if L.total_s() + cost > budget_s:
            break
        L.add("teleop", res.ticks * 0.1 + SETUP_S)
        L.add("reset", RESET_S)
        episodes.append(res)
        for s in res.steps:
            if s.placed and len(s.obs) > 3:
                segs.append(Segment(np.array(s.obs), np.array(s.act), "teleop", s.obj, {"card": cj["seed"]}))
        observed.append(_observed_from_episode(chore, cj, res))
    prog = infer_program(observed, CHORES[chore].objects) if observed else None
    return segs, L, episodes, prog


def _observed_from_episode(chore: str, card_json: dict, res: EpisodeResult) -> ObservedDemo:
    """Pick/place positions the robot camera saw in a teleop episode (start of step / after the step settles).
    Static objects use the card's start poses (the camera sees them at the start of the episode)."""
    poses = card_json["poses"]
    lib = {o.name: o for o in CHORES[chore].object_specs(card_json.get("subs") or None)}
    placed: dict[str, np.ndarray] = {}
    picks, at_place = [], []
    for s in res.steps:
        if s.final_perceived is None or s.perceived_obj is None:
            continue
        place = s.final_perceived.copy()
        place[2] = max(0.0, place[2] - lib[s.obj].height / 2)
        snap = {}
        for n, p in poses.items():
            if n == s.obj:
                continue
            snap[n] = placed.get(n, np.array([p[0], p[1], 0.0]))
        picks.append((s.obj, s.perceived_obj.copy(), place))
        at_place.append(snap)
        placed[s.obj] = place
    fixtures = {f: np.array([FIXTURES[f].pos[0], FIXTURES[f].pos[1], FIXTURES[f].surface_z])
                for f in CHORES[chore].fixtures}
    return ObservedDemo(picks, at_place, fixtures)


# ------------------------------------------------------------------------------------------ sim pre-training data
SIMPRE_OBJECTS = ["pen", "cup", "tape", "box", "can", "block_red", "block_blue", "plate_red", "snack_box", "eraser"]


def _simpre_episode(seed: int) -> list[Segment]:
    """Task-agnostic random pick-and-place in the digital twin with the scripted planner (D-019): random objects,
    random goals (a random point on the mat, inside a random bin, or on a shelf/tray), seeds disjoint from eval."""
    from sim.expert import PickPlaceTask, ScriptedPickPlace

    rng = np.random.default_rng(seed)
    objs = list(rng.choice(SIMPRE_OBJECTS, size=int(rng.integers(2, 4)), replace=False))
    fix = [str(rng.choice(["bin", "shelf"])), str(rng.choice(["bin_right", "tray"]))]
    goals: dict[str, PlaceGoal] = {}
    for o in objs:
        r = rng.random()
        if r < 0.35:   # somewhere on the mat (relative to the first fixture as a dummy landmark -> absolute later)
            goals[o] = PlaceGoal("mat", (float(rng.uniform(0.11, 0.25)), float(rng.uniform(-0.12, 0.12)), 0.0))
        else:
            f = FIXTURES[str(rng.choice(fix))]
            goals[o] = PlaceGoal(f"fixture:{f.name}", (float(rng.uniform(-0.5, 0.5) * f.half[0]),
                                                       float(rng.uniform(-0.5, 0.5) * f.half[1]), 0.0))
    chore = ChoreSpec("simpre", objs, fix, {})
    card = sample_card(chore, 500_000 + seed, "practice")
    env = make_env(chore, card)
    env.reset(card.placement)

    class _Expert:
        def begin_step(self, obj, obj_pos, obj_yaw, goal):
            self.c = ScriptedPickPlace(PickPlaceTask(obj, np.asarray(goal, float)), obj_pos, obj_yaw)

        def act(self, view):
            return self.c.act(view.ee_pos)

    out = []
    perc = OverheadPerceiver()
    for o in objs:
        g = goals[o]
        if g.landmark == "mat":
            spec = PlaceGoal(f"fixture:{fix[0]}", tuple(np.array(g.offset) - np.array(
                [FIXTURES[fix[0]].pos[0], FIXTURES[fix[0]].pos[1], FIXTURES[fix[0]].surface_z])))  # type: ignore[arg-type]
        else:
            spec = g
        rec = run_object_step(env, _Expert(), chore, o, spec, perc, record=True)
        p, _ = env.obj_pose(o)
        if rec.goal is not None and np.linalg.norm(p[:2] - rec.goal[:2]) < 0.03 and len(rec.obs) > 3:
            out.append(Segment(np.array(rec.obs), np.array(rec.act), "simpre", o))
    env.close()
    return out


def simpre_segments(n_episodes: int = 120) -> list[Segment]:
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(_simpre_episode, range(n_episodes)))
    return [s for r in res for s in r]


# ------------------------------------------------------------------------------------------ evaluation
_WORKER_STATE: dict = {}


def _init_worker(pol_bytes: list[bytes], agent_kw: dict) -> None:
    import io

    import torch

    torch.set_num_threads(1)
    from policies.data import Normalizer
    from policies.models import ACT, DiffusionPolicy  # noqa: F401
    from policies.train import TrainConfig, TrainedPolicy, make_model

    pols = []
    for b in pol_bytes:
        d = torch.load(io.BytesIO(b), weights_only=False)
        cfg = TrainConfig(**d["cfg"])
        m = make_model(cfg)
        m.load_state_dict(d["state"])
        m.eval()
        pols.append(TrainedPolicy(cfg, m, Normalizer.from_state(d["obs_norm"]), Normalizer.from_state(d["act_norm"]),
                                  d["data_hash"], 0, 0.0))
    _WORKER_STATE["pols"] = pols
    _WORKER_STATE["kw"] = agent_kw


def _eval_one(args) -> dict:
    chore_name, card_json, program, order, perceiver, seed, hook_name, hook_kw = args
    from policies.agent import PolicyAgent

    chore = CHORES[chore_name]
    card = PlacementCard.from_json(card_json)
    agent = PolicyAgent(_WORKER_STATE["pols"], seed=seed, **_WORKER_STATE["kw"])
    perc = _make_perceiver(perceiver, seed)
    env = make_env(chore, card, distractors=hook_kw.get("distractors"))
    hook = None
    if hook_name == "recovery":
        from recovery.recover import RecoveryHook

        hook = RecoveryHook(chore, perc, **{k: v for k, v in hook_kw.items() if k != "distractors"})
    res = run_episode(chore, card, agent, perc, program=program, order=order, env=env, step_hook=hook)
    out = episode_summary(res)
    out["uncertainty"] = [{"obj": t["obj"], "disagree": float(np.mean(t["disagree"])) if t["disagree"] else None,
                           "spread": float(np.mean(t["spread"])) if t["spread"] else None} for t in agent.trace]
    if hook is not None:
        out["recovery"] = hook.log
    env.close()
    return out


def _make_perceiver(name: str, seed: int):
    if name == "overhead":
        return OverheadPerceiver()
    if name.startswith("bump"):
        return OverheadPerceiver(calib_offset=np.array([float(name.split(":")[1]), 0.0, 0.0]))
    if name == "gt_noisy":
        return GTNoisyPerceiver(seed=seed)
    raise ValueError(name)


def episode_summary(res: EpisodeResult) -> dict:
    return {"seed": res.card["seed"], "split": res.card["split"], "success": bool(res.success),
            "n_done": res.n_done, "n_total": res.n_total, "ticks": res.ticks, "per_object": res.per_object,
            "phase": res.phase_rates(), "safety_events": res.safety_events,
            "steps": [{"obj": s.obj, "attempt": s.attempt, "reached": s.reached, "grasped": s.grasped,
                       "transported": s.transported, "placed": s.placed, "timeout": s.timeout, "ticks": s.ticks}
                      for s in res.steps]}


def evaluate(chore: str, pols, program: InferredProgram | None, split: str, n: int | None = None,
             perceiver: str = "overhead", agent_kw: dict | None = None, hook: str | None = None,
             hook_kw: dict | None = None, cards: list[PlacementCard] | None = None) -> list[dict]:
    """Run the frozen-protocol trials for `split` in parallel. `program=None` -> the agent is given the TRUE task
    program (used only for diagnostic upper bounds, never for headline results)."""
    cards = cards if cards is not None else load_protocol(chore, split)
    if n is not None:
        cards = cards[:n]
    prog = program.program if program is not None else None
    order = program.order if program is not None else None
    if program is not None and not prog:
        return [{"seed": c.seed, "split": c.split, "success": False, "n_done": 0, "n_total": len(CHORES[chore].program),
                 "ticks": 0, "per_object": {}, "phase": {}, "safety_events": [], "steps": [], "uncertainty": [],
                 "note": "empty inferred program"} for c in cards]
    jobs = [(chore, c.to_json(), prog, order, perceiver, 17 + i, hook, hook_kw or {}) for i, c in enumerate(cards)]
    blobs = [p.checkpoint_bytes() for p in pols]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(blobs, agent_kw or {})) as ex:
        return list(ex.map(_eval_one, jobs))


def rate(trials: list[dict], key: str = "success") -> dict:
    from eval.stats import wilson

    k = int(sum(bool(t[key]) for t in trials))
    n = len(trials)
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def object_rate(trials: list[dict]) -> dict:
    from eval.stats import wilson

    k = int(sum(t["n_done"] for t in trials))
    n = int(sum(t["n_total"] for t in trials))
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def cmdline() -> str:
    return " ".join(["python"] + sys.argv)


_ = object_at_goal
