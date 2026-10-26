"""Infer the chore's task program (which objects move, where to, in what order) from observed demos.

For each moved object, choose the landmark (a fixture, or another object's position at the time of
placing) that minimises the variance of the relative place position across demos; the goal is that
landmark plus the mean offset.  Objects never moved in any demo are static.  A pick order is enforced
only if it is identical in every demo, otherwise the robot picks nearest-first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval.chores import PlaceGoal


@dataclass
class ObservedDemo:
    picks: list[tuple[str, np.ndarray, np.ndarray]]          # (obj, pick_pos, place_pos) in pick order
    object_positions_at_place: list[dict[str, np.ndarray]]   # for each pick: other objects' positions then
    fixtures: dict[str, np.ndarray]                          # fixture centre (x, y, surface_z)


@dataclass
class InferredProgram:
    program: dict[str, PlaceGoal]
    order: list[str] | None
    static: list[str]
    spread: dict[str, float]          # std of relative place position (m) for the chosen landmark
    n_demos: int


def infer_program(demos: list[ObservedDemo], all_objects: list[str]) -> InferredProgram:
    rel: dict[str, dict[str, list[np.ndarray]]] = {}
    for d in demos:
        for k, (obj, _pick, place) in enumerate(d.picks):
            cands = {f"fixture:{n}": p for n, p in d.fixtures.items()}
            for o, p in d.object_positions_at_place[k].items():
                if o != obj:
                    cands[f"object:{o}"] = p
            for lm, lp in cands.items():
                rel.setdefault(obj, {}).setdefault(lm, []).append(place - lp)
    program: dict[str, PlaceGoal] = {}
    spread: dict[str, float] = {}
    for obj, by_lm in rel.items():
        n_obj = max(len(v) for v in by_lm.values())
        best = None
        for lm, vs in by_lm.items():
            if len(vs) < n_obj:  # landmark must be present in every demo of this object
                continue
            arr = np.array(vs)
            sd = float(np.sqrt(np.mean(np.var(arr[:, :2], axis=0)))) if len(arr) > 1 else 0.05
            score = sd + 0.02 * float(np.linalg.norm(arr.mean(axis=0)[:2]))  # tie-break: nearest landmark
            if best is None or score < best[3] - 1e-5:
                best = (sd, lm, arr.mean(axis=0), score)
        if best is None:
            continue
        sd, lm, off, _ = best
        if lm.startswith("fixture:"):
            off = np.array([off[0], off[1], 0.0])
        program[obj] = PlaceGoal(lm, (float(off[0]), float(off[1]), float(off[2])))
        spread[obj] = sd
    moved = set(program)
    static = [o for o in all_objects if o not in moved]
    order = infer_order([[o for o, _, _ in d.picks] for d in demos], program)
    return InferredProgram(program, order, static, spread, len(demos))


def infer_order(orders: list[list[str]], program: dict[str, PlaceGoal], min_demos: int = 5,
                consistency: float = 0.9) -> list[str] | None:
    """A pick order is enforced if (a) an object's goal landmark is another moved object (it must be placed
    first), or (b) every pair of objects appears in the same relative order in >= 90 % of >= 5 demos. Random
    pick orders (any order allowed) fail (b): for 2 objects, P(>= 9/10 agree by chance) ~ 2 %."""
    objs = list(program)
    before: dict[tuple[str, str], bool] = {}
    for a, g in program.items():
        kind, lm = g.landmark.split(":", 1)
        if kind == "object" and lm in program:
            before[(lm, a)] = True
    if len(orders) >= min_demos:
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                if (a, b) in before or (b, a) in before:
                    continue
                both = [o for o in orders if a in o and b in o]
                if len(both) < min_demos:
                    continue
                f = float(np.mean([o.index(a) < o.index(b) for o in both]))
                if f >= consistency:
                    before[(a, b)] = True
                elif f <= 1 - consistency:
                    before[(b, a)] = True
    if not before:
        return None
    # topological sort of the constrained objects (ties: object name, deterministic)
    constrained = sorted({x for pair in before for x in pair})
    out: list[str] = []
    left = set(constrained)
    while left:
        free = sorted(x for x in left if not any((y, x) in before for y in left if y != x))
        if not free:
            return None  # inconsistent constraints: do not enforce an order
        out.append(free[0])
        left.remove(free[0])
    return out
