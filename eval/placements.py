"""Seeded placement generator on the printed mat grid, plus the frozen evaluation protocol.

Grid: rows A..E along +x (A nearest the robot), columns 1..6 along +y (1 = robot's right).
HELD_OUT_CELLS are never used for filming/teleop/practice placements; "unseen placement" eval
configurations put at least one object in a held-out cell.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.chores import CHORES, ChoreSpec
from sim.env import Placement
from sim.scene import GRID_COLS, GRID_ROWS, LIB, MAT_X, MAT_Y

PROTOCOL_DIR = Path(__file__).resolve().parent / "protocol"
HELD_OUT_CELLS = ("A1", "C3", "E6", "B5", "D2")
ROTATIONS_DEG = (0, 45, 90, 135)
LIGHTING = {"normal": (1.0, (1.0, 1.0, 1.0)), "dim": (0.6, (1.0, 1.0, 1.0)),
            "bright_warm": (1.35, (1.0, 0.88, 0.72))}
# Seed spaces are disjoint by construction: eval seeds >= 1_000_000, training-side seeds < 1_000_000.
EVAL_SEED_BASE = 1_000_000


def cell_center(cell: str) -> tuple[float, float]:
    r = GRID_ROWS.index(cell[0])
    c = int(cell[1:]) - 1
    x = MAT_X[0] + r * (MAT_X[1] - MAT_X[0]) / (len(GRID_ROWS) - 1)
    y = MAT_Y[0] + c * (MAT_Y[1] - MAT_Y[0]) / (GRID_COLS - 1)
    return float(x), float(y)


ALL_CELLS = [f"{r}{c}" for r in GRID_ROWS for c in range(1, GRID_COLS + 1)]
TRAIN_CELLS = [c for c in ALL_CELLS if c not in HELD_OUT_CELLS]


def cell_of(xy: np.ndarray) -> str:
    best = min(ALL_CELLS, key=lambda c: (cell_center(c)[0] - xy[0]) ** 2 + (cell_center(c)[1] - xy[1]) ** 2)
    return best


def footprint_radius(name: str) -> float:
    o = LIB[name]
    if o.kind == "box":
        return float(np.hypot(o.size[0], o.size[1]))
    if o.kind == "cylinder":
        return float(o.size[0])
    return float(o.size[1] + o.size[0])


@dataclass
class PlacementCard:
    chore: str
    seed: int
    cells: dict[str, str]
    placement: Placement
    split: str  # film | teleop | practice | request | eval_seen | eval_unseen_place | eval_unseen_obj
    lighting: str = "normal"
    subs: dict[str, str] | None = None

    def to_json(self) -> dict:
        return {"chore": self.chore, "seed": self.seed, "cells": self.cells, "split": self.split,
                "lighting": self.lighting, "subs": self.subs or {},
                "poses": {k: [round(float(v), 5) for v in p] for k, p in self.placement.poses.items()}}

    @staticmethod
    def from_json(d: dict) -> PlacementCard:
        return PlacementCard(d["chore"], d["seed"], d["cells"], Placement({k: tuple(v) for k, v in d["poses"].items()}),
                             d["split"], d.get("lighting", "normal"), d.get("subs") or None)

    def human_text(self) -> str:
        parts = []
        for name, cell in self.cells.items():
            yaw = np.rad2deg(self.placement.poses[name][2]) % 180
            parts.append(f"{name} at {cell}, rotated {yaw:.0f}°")
        return "; ".join(parts)


def sample_card(chore: ChoreSpec, seed: int, split: str, cells_allowed: list[str] | None = None,
                force_cell: dict[str, str] | None = None, must_use_held_out: bool = False,
                subs: dict[str, str] | None = None, lighting: str = "normal", jitter: float = 0.008,
                fixed_rotation: dict[str, float] | None = None) -> PlacementCard:
    rng = np.random.default_rng(seed)
    allowed = list(cells_allowed or TRAIN_CELLS)
    names = chore.objects
    lib_names = {n: (subs or {}).get(n, n) for n in names}
    for _attempt in range(500):
        cells: dict[str, str] = {}
        poses: dict[str, tuple[float, float, float]] = {}
        order = list(names)
        rng.shuffle(order)
        ok = True
        for i, n in enumerate(order):
            if force_cell and n in force_cell:
                cell = force_cell[n]
            elif must_use_held_out and i == 0:
                cell = str(rng.choice(HELD_OUT_CELLS))
            else:
                cell = str(rng.choice(allowed))
            cx, cy = cell_center(cell)
            x, y = cx + rng.uniform(-jitter, jitter), cy + rng.uniform(-jitter, jitter)
            if fixed_rotation and n in fixed_rotation:
                yaw = fixed_rotation[n]
            else:
                yaw = np.deg2rad(float(rng.choice(ROTATIONS_DEG)) + rng.uniform(-10, 10))
            r_n = footprint_radius(lib_names[n])
            for f in chore.fixture_specs():
                if (abs(x - f.pos[0]) < f.half[0] + r_n + 0.012) and (abs(y - f.pos[1]) < f.half[1] + r_n + 0.012):
                    ok = False
            if not ok:
                break
            for m, (mx, my, _) in poses.items():
                if np.hypot(x - mx, y - my) < r_n + footprint_radius(lib_names[m]) + 0.012:
                    ok = False
                    break
            if not ok:
                break
            cells[n] = cell
            poses[n] = (float(x), float(y), float(yaw))
        if ok:
            return PlacementCard(chore.name, seed, cells, Placement(poses), split, lighting, subs)
    raise RuntimeError(f"could not sample a collision-free placement for {chore.name} seed={seed}")


def add_clutter(card: PlacementCard, names: list[str], seed: int,
                cells_allowed: list[str] | None = None) -> PlacementCard:
    """Copy of `card` with extra distractor objects (library names) placed on free mat cells (M7 clutter test).
    Collision-free w.r.t. the card's objects and the chore's fixtures; the card's own poses are unchanged."""
    chore = CHORES[card.chore]
    rng = np.random.default_rng(seed)
    poses = dict(card.placement.poses)
    cells = dict(card.cells)
    lib = {n: (card.subs or {}).get(n, n) for n in poses}
    for d in names:
        for _ in range(500):
            cell = str(rng.choice(cells_allowed or ALL_CELLS))
            cx, cy = cell_center(cell)
            x, y = cx + rng.uniform(-0.008, 0.008), cy + rng.uniform(-0.008, 0.008)
            r_d = footprint_radius(d)
            if any((abs(x - f.pos[0]) < f.half[0] + r_d + 0.012) and (abs(y - f.pos[1]) < f.half[1] + r_d + 0.012)
                   for f in chore.fixture_specs()):
                continue
            if any(np.hypot(x - mx, y - my) < r_d + footprint_radius(lib.get(m, m)) + 0.012
                   for m, (mx, my, _) in poses.items()):
                continue
            poses[d] = (float(x), float(y), float(np.deg2rad(rng.uniform(0, 180))))
            cells[d] = cell
            lib[d] = d
            break
        else:
            raise RuntimeError(f"no room for distractor {d}")
    return PlacementCard(card.chore, card.seed, cells, Placement(poses), card.split + "+clutter", card.lighting,
                         card.subs)


# ------------------------------------------------------------------------------------- frozen protocol
N_EVAL = {"eval_seen": 40, "eval_unseen_place": 40, "eval_unseen_obj": 20, "eval_lighting": 20}


def build_protocol(chore_name: str) -> list[PlacementCard]:
    chore = CHORES[chore_name]
    cards: list[PlacementCard] = []
    s = EVAL_SEED_BASE + 10_000 * list(CHORES).index(chore_name)
    for i in range(N_EVAL["eval_seen"]):
        cards.append(sample_card(chore, s + i, "eval_seen"))
    for i in range(N_EVAL["eval_unseen_place"]):
        cards.append(sample_card(chore, s + 1000 + i, "eval_unseen_place", cells_allowed=ALL_CELLS,
                                 must_use_held_out=True))
    if chore.novel_objects:
        for i in range(N_EVAL["eval_unseen_obj"]):
            cards.append(sample_card(chore, s + 2000 + i, "eval_unseen_obj", subs=chore.novel_objects))
    for i in range(N_EVAL["eval_lighting"]):
        light = ["dim", "bright_warm"][i % 2]
        cards.append(sample_card(chore, s + 3000 + i, "eval_lighting", lighting=light))
    return cards


def protocol_path(chore_name: str) -> Path:
    return PROTOCOL_DIR / f"{chore_name}.json"


def protocol_checksum(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def freeze_protocol(chore_name: str) -> str:
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps([c.to_json() for c in build_protocol(chore_name)], indent=1)
    protocol_path(chore_name).write_text(text)
    return protocol_checksum(text)


def load_protocol(chore_name: str, split: str | None = None, verify: bool = True) -> list[PlacementCard]:
    text = protocol_path(chore_name).read_text()
    if verify:
        sums = json.loads((PROTOCOL_DIR / "CHECKSUMS.json").read_text())
        if sums.get(chore_name) != protocol_checksum(text):
            raise RuntimeError(f"evaluation protocol for {chore_name} changed after freezing")
    cards = [PlacementCard.from_json(d) for d in json.loads(text)]
    return [c for c in cards if split is None or c.split == split]


def training_cards(chore_name: str, split: str, n: int, offset: int = 0) -> list[PlacementCard]:
    """Placements for filming / teleop / practice / candidate pools. Seeds disjoint from eval seeds."""
    chore = CHORES[chore_name]
    base = {"film": 0, "teleop": 100_000, "practice": 200_000, "request": 300_000, "pool": 400_000}[split]
    base += 10_000 * list(CHORES).index(chore_name)
    return [sample_card(chore, base + offset + i, split) for i in range(n)]


if __name__ == "__main__":
    sums = {name: freeze_protocol(name) for name in CHORES}
    (PROTOCOL_DIR / "CHECKSUMS.json").write_text(json.dumps(sums, indent=1))
    print(json.dumps(sums, indent=1))
