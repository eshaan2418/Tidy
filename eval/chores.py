"""Chore definitions: objects, fixtures, ground-truth task programs (used ONLY by the simulated human /
teleoperator and by scoring), success predicates (full-task and per-phase).

The learned system never reads `program`; it infers the task program from videos (phases/program.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from sim.scene import FIXTURES, LIB, FixtureSpec, ObjectSpec


@dataclass(frozen=True)
class PlaceGoal:
    """Where an object should end up: landmark position + offset. landmark = 'fixture:<name>' | 'object:<name>'."""
    landmark: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ChoreSpec:
    name: str
    objects: list[str]
    fixtures: list[str]
    program: dict[str, PlaceGoal]           # object -> goal (objects not listed stay put)
    order: list[str] | None = None          # required order (stacking); None = any
    static: list[str] = field(default_factory=list)   # objects that must not be moved
    novel_objects: dict[str, str] = field(default_factory=dict)  # unseen-object eval substitutions
    description: str = ""

    def object_specs(self, subs: dict[str, str] | None = None) -> list[ObjectSpec]:
        # a substituted (unseen) object keeps the chore's slot name so the task program still refers to it
        subs = subs or {}
        return [replace(LIB[subs[n]], name=n) if n in subs else LIB[n] for n in self.objects]

    def fixture_specs(self) -> list[FixtureSpec]:
        return [FIXTURES[f] for f in self.fixtures]

    def movable(self) -> list[str]:
        return [o for o in self.objects if o in self.program]


CHORES: dict[str, ChoreSpec] = {
    "desk_clearing": ChoreSpec(
        "desk_clearing", ["pen", "cup", "tape", "box"], ["bin"],
        {"cup": PlaceGoal("fixture:bin", (-0.040, -0.025, 0.0)), "tape": PlaceGoal("fixture:bin", (0.035, -0.025, 0.0)),
         "box": PlaceGoal("fixture:bin", (-0.030, 0.030, 0.0)), "pen": PlaceGoal("fixture:bin", (0.030, 0.030, 0.0))},
        novel_objects={"cup": "can", "box": "snack_box"},
        description="Put every item on the mat into the bin."),
    "sorting": ChoreSpec(
        "sorting", ["block_red", "block_blue", "block_red2", "block_blue2"], ["bin_left", "bin_right"],
        {"block_red": PlaceGoal("fixture:bin_left", (0.0, -0.025, 0.0)),
         "block_red2": PlaceGoal("fixture:bin_left", (0.0, 0.025, 0.0)),
         "block_blue": PlaceGoal("fixture:bin_right", (0.0, 0.025, 0.0)),
         "block_blue2": PlaceGoal("fixture:bin_right", (0.0, -0.025, 0.0))},
        description="Red blocks into the left bin, blue blocks into the right bin."),
    "shelf_putaway": ChoreSpec(
        "shelf_putaway", ["can", "snack_box", "cup"], ["shelf"],
        {"can": PlaceGoal("fixture:shelf", (0.020, -0.028, 0.0)),
         "snack_box": PlaceGoal("fixture:shelf", (-0.035, -0.020, 0.0)),
         "cup": PlaceGoal("fixture:shelf", (0.0, 0.032, 0.0))},
        description="Put the three items upright onto the shelf."),
    "stacking": ChoreSpec(
        "stacking", ["plate_blue", "plate_red", "plate_yellow"], [],
        {"plate_red": PlaceGoal("object:plate_blue", (0.0, 0.0, 0.020)),
         "plate_yellow": PlaceGoal("object:plate_red", (0.0, 0.0, 0.020))},
        order=["plate_red", "plate_yellow"], static=["plate_blue"],
        description="Leave the blue plate; stack red on blue, then yellow on red."),
    "tray_heldout": ChoreSpec(
        "tray_heldout", ["pen", "eraser", "cup"], ["tray"],
        {"pen": PlaceGoal("fixture:tray", (0.0, 0.015, 0.0)), "eraser": PlaceGoal("fixture:tray", (0.0, -0.02, 0.0))},
        static=["cup"],
        description="HELD-OUT: put the pen and the eraser onto the tray; leave the cup."),
}


def landmark_position(landmark: str, env) -> np.ndarray:
    kind, name = landmark.split(":", 1)
    if kind == "fixture":
        f = env.fix_by_name[name]
        return np.array([f.pos[0], f.pos[1], f.surface_z])
    p, _ = env.obj_pose(name)
    o = env.obj_by_name[name]
    return np.array([p[0], p[1], p[2] - o.height / 2])  # base of the object


def goal_position(goal: PlaceGoal, env) -> np.ndarray:
    return landmark_position(goal.landmark, env) + np.array(goal.offset)


# ------------------------------------------------------------------------------------- success predicates
def object_at_goal(chore: ChoreSpec, name: str, env, goal: PlaceGoal | None = None) -> bool:
    goal = goal or chore.program[name]
    p, _ = env.obj_pose(name)
    o = env.obj_by_name[name]
    kind, lname = goal.landmark.split(":", 1)
    if kind == "fixture":
        f = env.fix_by_name[lname]
        if f.kind == "bin":
            return f.contains_xy(p[:2]) and p[2] < f.height + 0.10
        # shelf / tray: resting on the top surface inside the region, upright
        on_top = p[2] - o.height / 2 > f.height - 0.01
        return f.contains_xy(p[:2]) and on_top and env.obj_upright(name)
    # object-relative (stacking): centred on the landmark object within 2.2 cm, above it, flat
    lp, _ = env.obj_pose(lname)
    lo = env.obj_by_name[lname]
    above = p[2] - o.height / 2 > lp[2] + lo.height / 2 - 0.008
    return bool(np.linalg.norm(p[:2] - lp[:2]) < 0.022 and above and env.obj_upright(name))


def chore_success(chore: ChoreSpec, env, initial: dict[str, np.ndarray] | None = None) -> dict:
    per_obj = {n: object_at_goal(chore, n, env) for n in chore.movable()}
    static_ok = True
    if initial:
        for s in chore.static:
            static_ok &= bool(np.linalg.norm(env.obj_pose(s)[0][:2] - initial[s][:2]) < 0.02)
    return {"success": all(per_obj.values()) and static_ok, "per_object": per_obj, "static_ok": static_ok,
            "n_done": int(sum(per_obj.values())), "n_total": len(per_obj)}


PHASE_NAMES = ("reach", "grasp", "transport", "place")
