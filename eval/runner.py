"""Chore executor shared by every agent (simulated teleoperator, learned policies, recovery wrapper).

Task level: the executor walks the task program (inferred from videos for learned agents) object
by object.  For each object step it perceives the scene once at step start (perceive-then-act;
the same convention is used when building demos), then runs the agent at 10 Hz until it releases
the object (+3 ticks) or times out.  Per-phase outcomes are scored from simulator ground truth
(sim stage) -- on the real robot the same fields come from the overhead-camera scorer + human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from common.geometry import wrap_half_pi
from eval.chores import ChoreSpec, PlaceGoal, chore_success, landmark_position, object_at_goal
from eval.placements import LIGHTING, PlacementCard
from policies.spec import View, action_vector, obs_vector
from sim.env import EEAction, TidyEnv
from sim.scene import ObjectSpec

STEP_TIMEOUT = 220  # ticks (22 s) per object step


# ------------------------------------------------------------------------------------------ perception
class Perceiver(Protocol):
    def perceive(self, env: TidyEnv, names: list[str]) -> dict[str, tuple[np.ndarray, float]]: ...


class GTNoisyPerceiver:
    """Ground truth + Gaussian noise. Default sigmas are set from the measured sim perception error (M3)."""

    def __init__(self, sigma_pos: float = 0.004, sigma_yaw: float = np.deg2rad(4), seed: int = 0):
        self.sp, self.sy = sigma_pos, sigma_yaw
        self.rng = np.random.default_rng(seed)

    def perceive(self, env: TidyEnv, names: list[str]) -> dict[str, tuple[np.ndarray, float]]:
        out = {}
        for n in names:
            p, y = env.obj_pose(n)
            noise = self.rng.normal(0, self.sp, 3)
            noise[2] *= 0.3
            out[n] = (p + noise, float(wrap_half_pi(y + self.rng.normal(0, self.sy))))
        return out


# ------------------------------------------------------------------------------------------ agents
class Agent(Protocol):
    def begin_step(self, obj: ObjectSpec, obj_pos: np.ndarray, obj_yaw: float, goal: np.ndarray) -> None: ...
    def act(self, view: View) -> EEAction: ...


@dataclass
class StepRecord:
    obj: str
    obs: list[np.ndarray] = field(default_factory=list)
    act: list[np.ndarray] = field(default_factory=list)
    reached: bool = False
    grasped: bool = False
    transported: bool = False
    placed: bool = False
    ticks: int = 0
    timeout: bool = False
    attempt: int = 0
    min_xy_dist_open: float = 1.0
    perceived_obj: np.ndarray | None = None
    goal: np.ndarray | None = None
    release_xy_err: float | None = None
    frames: list[np.ndarray] = field(default_factory=list)


@dataclass
class EpisodeResult:
    success: bool
    n_done: int
    n_total: int
    steps: list[StepRecord]
    ticks: int
    safety_events: list[str]
    per_object: dict[str, bool]
    card: dict

    def phase_rates(self) -> dict[str, float]:
        if not self.steps:
            return {}
        keys = ("reached", "grasped", "transported", "placed")
        return {k: float(np.mean([getattr(s, k) for s in self.steps])) for k in keys}


def make_env(chore: ChoreSpec, card: PlacementCard, include_hand: bool = False) -> TidyEnv:
    light, tint = LIGHTING[card.lighting]
    return TidyEnv(chore.object_specs(card.subs), chore.fixture_specs(), include_hand=include_hand,
                   light_scale=light, light_tint=tint)


def make_view(env: TidyEnv, obj: ObjectSpec, obj_pos: np.ndarray, obj_yaw: float, goal: np.ndarray) -> View:
    ee, yaw, pitch = env.ee_state()
    return View(ee, yaw, pitch, env.opening(), env.g_cmd, obj_pos, obj_yaw, obj, goal)


def release_in_goal_region(spec: PlaceGoal, p: np.ndarray, goal: np.ndarray, env: TidyEnv) -> bool:
    kind, name = spec.landmark.split(":", 1)
    if kind == "fixture" and env.fix_by_name[name].kind == "bin":
        return env.fix_by_name[name].contains_xy(p[:2], margin=0.005)
    return float(np.linalg.norm(p[:2] - goal[:2])) < 0.02


def run_object_step(env: TidyEnv, agent: Agent, chore: ChoreSpec, name: str, goal_spec: PlaceGoal,
                    perceiver: Perceiver, record: bool = False, timeout: int = STEP_TIMEOUT,
                    render_every: int = 0, monitor=None) -> StepRecord:
    obj = env.obj_by_name[name]
    seen = perceiver.perceive(env, [name] + ([goal_spec.landmark.split(":", 1)[1]]
                                             if goal_spec.landmark.startswith("object:") else []))
    op, oy = seen[name]
    if goal_spec.landmark.startswith("object:"):
        ln = goal_spec.landmark.split(":", 1)[1]
        lp, _ = seen[ln]
        goal = np.array([lp[0], lp[1], lp[2] - env.obj_by_name[ln].height / 2]) + np.array(goal_spec.offset)
    else:
        goal = landmark_position(goal_spec.landmark, env) + np.array(goal_spec.offset)
    rec = StepRecord(name, perceived_obj=op.copy(), goal=goal.copy())
    true_spec = chore.program.get(name)
    true_goal = None
    if true_spec is not None:  # scoring uses the TRUE task program and GT poses, never the inferred one
        from eval.chores import goal_position

        true_goal = goal_position(true_spec, env)
    agent.begin_step(obj, op, oy, goal)
    released_at = None
    ever_attached = False
    for t in range(timeout):
        v = make_view(env, obj, op, oy, goal)
        a = agent.act(v)
        if record:
            rec.obs.append(obs_vector(v))
            rec.act.append(action_vector(v.ee_pos, a))
        true_p, _ = env.obj_pose(name)
        if env.g_cmd < 0.5 and not ever_attached:
            rec.min_xy_dist_open = min(rec.min_xy_dist_open, float(np.linalg.norm(v.ee_pos[:2] - true_p[:2])))
        info = env.step(a, monitor_rgb=monitor(env) if monitor else None)
        if render_every and t % render_every == 0:
            rec.frames.append(env.render("overhead", 160, 120))
        if env.attached == name:
            ever_attached = True
            if true_p[2] > obj.height / 2 + 0.03:
                rec.grasped = True
        if info["released"] == name or (ever_attached and env.attached != name and released_at is None):
            released_at = t
            p_rel, _ = env.obj_pose(name)
            if true_goal is not None and true_spec is not None:
                rec.release_xy_err = float(np.linalg.norm(p_rel[:2] - true_goal[:2]))
                rec.transported = release_in_goal_region(true_spec, p_rel, true_goal, env)
        if released_at is not None and t - released_at >= 3:
            break
        rec.ticks = t + 1
    else:
        rec.timeout = True
    rec.reached = rec.min_xy_dist_open < 0.02 or ever_attached
    env.settle(60)
    rec.placed = true_spec is not None and object_at_goal(chore, name, env)
    return rec


def order_objects(chore: ChoreSpec, env: TidyEnv, program: dict[str, PlaceGoal], remaining: list[str],
                  order: list[str] | None) -> str:
    if order:
        for n in order:
            if n in remaining:
                return n
    ee, _, _ = env.ee_state()
    return min(remaining, key=lambda n: float(np.linalg.norm(env.obj_pose(n)[0][:2] - ee[:2])))


def run_episode(chore: ChoreSpec, card: PlacementCard, agent: Agent, perceiver: Perceiver,
                program: dict[str, PlaceGoal] | None = None, order: list[str] | None = None,
                record: bool = False, env: TidyEnv | None = None, step_hook=None,
                render_every: int = 0) -> EpisodeResult:
    """Run one full chore. `step_hook(env, rec) -> 'retry'|'next'|'help'` enables recovery (M6)."""
    program = program if program is not None else chore.program
    order = order if order is not None else chore.order
    env = env or make_env(chore, card)
    env.reset(card.placement)
    initial = {n: env.obj_pose(n)[0] for n in chore.objects}
    remaining = [n for n in chore.objects if n in program]
    steps: list[StepRecord] = []
    while remaining:
        name = order_objects(chore, env, program, remaining, order)
        attempt = 0
        while True:
            rec = run_object_step(env, agent, chore, name, program[name], perceiver, record=record,
                                  render_every=render_every)
            rec.attempt = attempt
            steps.append(rec)
            decision = step_hook(env, rec) if step_hook else "next"
            if decision == "retry":
                attempt += 1
                continue
            break
        remaining.remove(name)
    succ = chore_success(chore, env, initial)
    total_ticks = sum(s.ticks for s in steps)
    return EpisodeResult(succ["success"], succ["n_done"], succ["n_total"], steps, total_ticks,
                         list(env.safety_events), succ["per_object"], card.to_json())
