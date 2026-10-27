"""Uncertainty-driven demo requests (Contribution 2).

The robot asks for the single most useful extra demonstration. Signals (all available on the real robot):
  * ensemble disagreement: std across bootstrap-ensemble members of the next TCP target, averaged over an object
    step, split by phase (pick = jaw open, place = jaw closed). Computed by rolling the ensemble out on a
    candidate placement in the digital twin (the robot "imagines" doing the chore there).
  * action-distribution spread (Diffusion Policy samples), when a DP policy is used.
  * predicted failure: the per-phase failure detector (recovery/) applied to the imagined step.
Request = the candidate placement (from the mat grid) whose worst object step has the highest score. The text names
that object, its cell and rotation, and the uncertain phase, and a key frame shows the scene with the object marked.

Baselines at equal demo count / human time: `random` (a random candidate placement) and `round_robin` (cycles
through objects x mat regions for coverage).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.chores import CHORES
from eval.placements import GRID_ROWS, TRAIN_CELLS, PlacementCard, sample_card

PHASE_TEXT = {"pick": "pick it up", "place": "put it in place"}
REQUEST_SEED_BASE = 300_000   # request split (training-side seeds, disjoint from eval)


@dataclass
class DemoRequest:
    strategy: str
    card: dict
    obj: str
    cell: str
    rotation_deg: float
    phase: str
    score: float
    text: str
    key_frame: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def request_text(obj: str, cell: str, rot: float | None, phase: str, others: str) -> str:
    nice = obj.replace("_", " ")
    turn = f", rotated {rot:.0f}°" if rot is not None else ""
    return (f"Place the {nice} at {cell}{turn}, and show me once more how you {PHASE_TEXT[phase]}. "
            f"(Other items: {others}.)")


def describe(card: PlacementCard, obj: str, strategy: str, phase: str, score: float) -> DemoRequest:
    from sim.scene import LIB

    rot = float(np.rad2deg(card.placement.poses[obj][2]) % 180)
    lib = LIB[(card.subs or {}).get(obj, obj)]
    others = "; ".join(f"{n.replace('_', ' ')} at {c}" for n, c in card.cells.items() if n != obj) or "none"
    return DemoRequest(strategy, card.to_json(), obj, card.cells[obj], rot, phase, score,
                       request_text(obj, card.cells[obj], None if lib.kind == "cylinder" else rot, phase, others))


def candidate_pool(chore: str, n: int, round_idx: int, seed: int) -> list[PlacementCard]:
    """Fresh candidate placements per round (training-side seeds; never eval placements)."""
    base = REQUEST_SEED_BASE + 10_000 * list(CHORES).index(chore) + 1000 * seed + 100 * round_idx
    return [sample_card(CHORES[chore], base + i, "request") for i in range(n)]


def score_candidates(trials: list[dict], chore: str) -> list[tuple[float, str, str]]:
    """Per candidate trial (from bench.common.evaluate on the pool): (score, object, phase) of its worst step.
    Score = mean ensemble disagreement (m) of the step's most uncertain phase."""
    out = []
    for t in trials:
        best = (-1.0, "", "pick")
        for u in t.get("uncertainty", []):
            for ph in ("pick", "place"):
                v = u.get(f"disagree_{ph}")
                if v is not None and v > best[0]:
                    best = (float(v), u["obj"], ph)
        if best[1] == "":   # no ensemble trace (e.g. empty program): no information, lowest priority
            best = (0.0, CHORES[chore].movable()[0], "pick")
        out.append(best)
    return out


def pick_uncertain(pool: list[PlacementCard], trials: list[dict], k: int, chore: str) -> list[DemoRequest]:
    scored = score_candidates(trials, chore)
    order = np.argsort([-s for s, _, _ in scored])
    return [describe(pool[i], scored[i][1], "uncertainty", scored[i][2], scored[i][0]) for i in order[:k]]


def pick_random(pool: list[PlacementCard], k: int, rng: np.random.Generator) -> list[DemoRequest]:
    idx = rng.choice(len(pool), size=k, replace=False)
    return [describe(pool[i], str(rng.choice(list(pool[i].cells))), "random", str(rng.choice(["pick", "place"])), 0.0)
            for i in idx]


def coverage_cells() -> list[str]:
    """Training cells ordered to sweep the mat: alternate far/near rows and left/right columns."""
    rows = [GRID_ROWS[i] for i in (0, 4, 2, 1, 3) if i < len(GRID_ROWS)]
    out = []
    for j, r in enumerate(rows):
        cols = sorted({int(c[1:]) for c in TRAIN_CELLS if c[0] == r})
        cols = cols if j % 2 == 0 else cols[::-1]
        out += [f"{r}{c}" for c in cols if f"{r}{c}" in TRAIN_CELLS]
    return out


def pick_round_robin(chore: str, counter: int, k: int, seed: int) -> list[DemoRequest]:
    """Request j: object j mod n_objects, placed in coverage cell j // n_objects (others random)."""
    spec = CHORES[chore]
    movable = spec.movable()
    cells = coverage_cells()
    out = []
    for j in range(counter, counter + k):
        obj = movable[j % len(movable)]
        cell = cells[(j // len(movable) * 7) % len(cells)]
        card = sample_card(spec, REQUEST_SEED_BASE + 500_000 + 1000 * seed + j, "request", force_cell={obj: cell})
        out.append(describe(card, obj, "round_robin", "pick" if j % 2 == 0 else "place", 0.0))
    return out


def render_key_frame(req: DemoRequest, path: Path) -> str:
    """Overhead render of the requested placement with the requested object circled."""
    import cv2

    from eval.runner import make_env
    from perception.overhead import CameraModel

    chore = CHORES[req.card["chore"]]
    card = PlacementCard.from_json(req.card)
    env = make_env(chore, card)
    env.reset(card.placement)
    W, H = 320, 240
    rgb = env.render("overhead", W, H).copy()
    cam = CameraModel.from_mujoco(env.model, env.data, "overhead", W, H)
    p, _ = env.obj_pose(req.obj)
    pc = cam.R @ p + cam.t
    u, v = cam.K[0, 0] * pc[0] / pc[2] + cam.K[0, 2], cam.K[1, 1] * pc[1] / pc[2] + cam.K[1, 2]
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(img, (int(u), int(v)), 22, (0, 0, 255), 2)
    cv2.putText(img, f"{req.obj} @ {req.cell}: {req.phase}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    env.close()
    req.key_frame = str(path)
    return str(path)


def save_requests(reqs: list[DemoRequest], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_json() for r in reqs], indent=1))
