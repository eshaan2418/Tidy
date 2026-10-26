import json

import numpy as np

from eval.chores import CHORES
from eval.placements import (
    EVAL_SEED_BASE,
    HELD_OUT_CELLS,
    PROTOCOL_DIR,
    cell_of,
    load_protocol,
    protocol_checksum,
    protocol_path,
    training_cards,
)
from eval.stats import newcombe_diff, wilson


def test_wilson_known_values():
    p, lo, hi = wilson(8, 10)
    assert abs(p - 0.8) < 1e-9 and abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    assert wilson(0, 20)[1] == 0.0
    d, lo, hi = newcombe_diff(18, 20, 10, 20)
    assert lo < d < hi


def test_protocols_frozen_and_checksummed():
    sums = json.loads((PROTOCOL_DIR / "CHECKSUMS.json").read_text())
    for name in CHORES:
        assert sums[name] == protocol_checksum(protocol_path(name).read_text())
        cards = load_protocol(name)
        assert all(c.seed >= EVAL_SEED_BASE for c in cards)
        seen = [c for c in cards if c.split == "eval_seen"]
        assert len(seen) >= 20
        assert all(cell not in HELD_OUT_CELLS for c in seen for cell in c.cells.values())
        unseen = [c for c in cards if c.split == "eval_unseen_place"]
        assert all(any(cell in HELD_OUT_CELLS for cell in c.cells.values()) for c in unseen)


def test_training_placements_never_use_eval_seeds_or_held_out_cells():
    for split in ("film", "teleop", "practice", "request", "pool"):
        for c in training_cards("desk_clearing", split, 30):
            assert c.seed < EVAL_SEED_BASE
            for x, y, _ in c.placement.poses.values():
                assert cell_of(np.array([x, y])) not in HELD_OUT_CELLS
