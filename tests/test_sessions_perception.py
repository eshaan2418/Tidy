import numpy as np

from eval.placements import load_protocol


def test_session_trials_balanced_and_shuffled(tmp_path, monkeypatch):
    from eval import queue

    monkeypatch.setattr(queue, "CKPT_DIR", tmp_path)
    progs = {c: {"program": {}, "order": None} for c in ("a", "b")}
    spec = queue.build_session("desk_clearing", "T", ["a", "b"], 4, 3, 0, progs)
    conds = [t["condition"] for t in spec["trials"]]
    assert conds.count("a") == conds.count("b") == 7
    seeds_a = sorted(t["card"]["seed"] for t in spec["trials"] if t["condition"] == "a")
    seeds_b = sorted(t["card"]["seed"] for t in spec["trials"] if t["condition"] == "b")
    assert seeds_a == seeds_b                                   # identical placements for every method
    assert conds != sorted(conds)                               # interleaved, not blocked by method


def test_rayplane_perceiver_close_to_truth():
    from eval.chores import CHORES
    from eval.runner import make_env
    from perception.overhead import RayPlanePerceiver

    chore = CHORES["desk_clearing"]
    card = load_protocol("desk_clearing", "eval_seen")[0]
    env = make_env(chore, card)
    env.reset(card.placement)
    seen = RayPlanePerceiver().perceive(env, chore.objects)
    errs = [np.linalg.norm(seen[n][0][:2] - env.obj_pose(n)[0][:2]) for n in chore.objects]
    env.close()
    assert max(errs) < 0.015, errs


def test_board_to_table_extrinsics_projects_consistently():
    import cv2

    from hardware.camera_calib import BOARD_ORIGIN, R_TABLE_BOARD, table_extrinsics

    rvec = np.array([2.9, 0.1, -0.05])
    tvec = np.array([0.02, -0.03, 0.55])
    R_cb, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    R, t = table_extrinsics(rvec, tvec)
    p_board = np.array([0.05, 0.03, 0.0])
    p_table = R_TABLE_BOARD @ p_board + BOARD_ORIGIN
    assert np.allclose(R_cb @ p_board + tvec, R @ p_table + t)
    assert np.allclose(R @ R.T, np.eye(3))


def test_real_clip_format_roundtrip(tmp_path):
    from eval.chores import CHORES
    from eval.placements import training_cards
    from perception.real_video import process_clip, write_clip
    from sim.human_demo import record_demo

    card = training_cards("desk_clearing", "film", 1)[0]
    v = record_demo(CHORES["desk_clearing"], card, seed=7)
    write_clip(tmp_path / "clip_00", v.rgb, v.K, v.fps, v.depth)
    out = process_clip("desk_clearing", tmp_path / "clip_00")
    assert out["accepted"], out["reasons"]
    # plumbing check only: hands come from the weak fingertip-cap detector here (M3 sweep), so not all 4 picks
    assert len(out["observed"].picks) >= 1 and out["duration_s"] > 5
