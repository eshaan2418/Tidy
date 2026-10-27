import numpy as np

from active.requests import coverage_cells, describe, pick_round_robin, score_candidates
from eval.placements import EVAL_SEED_BASE, HELD_OUT_CELLS, training_cards
from recovery.recover import FailureDetector, Logistic


def test_score_candidates_takes_worst_phase():
    trials = [{"uncertainty": [{"obj": "cup", "disagree_pick": 0.001, "disagree_place": 0.004},
                               {"obj": "pen", "disagree_pick": 0.002, "disagree_place": None}]},
              {"uncertainty": []}]
    s = score_candidates(trials, "desk_clearing")
    assert s[0] == (0.004, "cup", "place")
    assert s[1][0] == 0.0


def test_request_text_and_pool_disjoint_from_eval():
    from active.requests import candidate_pool

    pool = candidate_pool("desk_clearing", 5, 0, 0)
    assert all(c.seed < EVAL_SEED_BASE for c in pool)
    r = describe(pool[0], "box", "uncertainty", "pick", 0.01)
    assert r.cell in r.text and "rotated" in r.text and "pick it up" in r.text
    r2 = describe(pool[0], "cup", "uncertainty", "place", 0.01)
    assert "rotated" not in r2.text      # cylinders: rotation is meaningless


def test_round_robin_cycles_objects_and_avoids_held_out_cells():
    reqs = pick_round_robin("desk_clearing", 0, 8, 0)
    assert len({q.obj for q in reqs}) == 4
    assert not set(coverage_cells()) & set(HELD_OUT_CELLS)
    assert all(q.cell not in HELD_OUT_CELLS for q in reqs)


def test_training_cards_never_use_held_out_cells():
    for c in training_cards("desk_clearing", "film", 20):
        assert not set(c.cells.values()) & set(HELD_OUT_CELLS)


def test_detector_learns_and_calibrates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y = (X[:, 3] < 0.2).astype(float)          # "placed" iff distance-to-goal feature small
    det = FailureDetector.fit(X[:300], y[:300], y[:300], X[300:], y[300:], target_precision=0.8)
    flags = np.array([det.judge(x[None])[0] for x in X[300:]])
    fail = y[300:] < 0.5
    assert (flags & fail).sum() / max(1, flags.sum()) >= 0.8
    assert (flags & fail).sum() / fail.sum() > 0.7
    L = Logistic.from_json(det.place.to_json())
    assert np.allclose(L.proba(X[:5]), det.place.proba(X[:5]))


def test_recovery_hook_retries_then_asks_for_help():
    from types import SimpleNamespace

    from recovery.recover import RecoveryHook

    class AlwaysFail:
        def judge(self, x):
            return True, "grasp", 0.1

    calls = []

    class Env:
        obj_by_name = {"cup": __import__("sim.scene").scene.LIB["cup"]}

        def ee_state(self):
            return np.array([0.15, 0.0, 0.05]), 0.0, 0.3

        def step(self, a):
            calls.append(a)

    hook = RecoveryHook(None, None, detector=AlwaysFail(), max_retries=2)  # type: ignore[arg-type]
    rec = SimpleNamespace(obj="cup", attempt=0, placed=False, grasped=False, grip_closed_open=0.0, lifted_ticks=0,
                          timeout=True, final_perceived=np.array([0.2, 0, 0.03]),
                          perceived_obj=np.array([0.2, 0, 0.03]),
                          goal=np.array([0.04, 0.2, 0.045]), ticks=220)
    assert hook(Env(), rec) == "retry"
    rec.attempt = 2
    assert hook(Env(), rec) == "help"
    assert calls and all(c.gripper < 0.5 for c in calls)   # safe retreat opens the jaw


def test_claim_scanner_flags_hand_typed_numbers():
    from bench.check_claims import scan_template

    assert scan_template("success was {{m4.video_10.all}} (sim)") == []
    assert scan_template("we got 45% [33–57] on 17/40 trials") != []
    assert scan_template("see `results/sim/x_0.25.json` and [link](a/1/2)") == []
