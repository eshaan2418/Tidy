# PROGRESS

Evidence stage: **sim only** (no arm, no NVIDIA GPU, no phone videos yet; see BLOCKERS.md B-001…B-006). Numbers live in `results/` and `results/INDEX.md`; this file holds status only.

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap + prior work | done | `docs/PRIOR_WORK.md` (verified citations, novelty position), `ENVIRONMENT.md`, CI (`.github/workflows/ci.yml`), `Dockerfile`, `requirements.lock` |
| M1 Hardware + safety layer | done (sim) / awaiting-human (real S1) | `safety/`, `tests/test_safety.py`, `results/sim/safety_audit.json`, `hardware/CHECKLIST.md`, `hardware/camera_calib.py`, `hardware/latency.py`, `SESSION_QUEUE.md` S1 |
| M2 Eval protocol + teleop baseline | done (sim) / awaiting-human (real T1, S2) | `eval/protocol/` (checksummed), `eval/placements.py`, `eval/stats.py`, `hardware/mat.pdf`, teleop rows in `results/sim/m4_video_vs_teleop_desk_clearing.json` |
| M3 Human video pipeline | done (sim) / awaiting-human (real labels H6) | `results/sim/m3_perception.json` |
| M4 Retargeting + policy learning | done (sim) / awaiting-human (S2, P1) | `results/sim/m4_video_vs_teleop_desk_clearing.json`, `results/sim/dev_policy_design_desk_clearing.json` |
| M5 Uncertainty-driven requests | done (sim; negative result) / awaiting-human (S3) | `results/sim/m5_active_desk_clearing.json`, `results/sim/m5_requests/` |
| M6 Failure detection + recovery | done (sim) / awaiting-human (S4) | `results/sim/m6_recovery_desk_clearing.json`, `results/sim/m6_detector_desk_clearing.json` |
| M7 Multi-chore + generalization | done (sim); towel folding blocked-with-fallback (B-004) / awaiting-human (S5, S6) | `results/sim/m7_multichore.json` |
| M8 Teaching app | done (sim walkthrough) / awaiting-human (H9) | `scripts/teach.py`, `results/sim/m8_teach_tray_heldout.json` |
| M9 Integration + write-up | done (sim) | `make reproduce-quick`, `paper/paper.md`, `README.md`, `DEMO.md`, `RESUME_BULLETS.md`, `FINAL_REVIEW.md`, `bench/check_claims.py`, `bench/repo_scan.py` |

## Plan executed in this session (M2–M9)
1. [x] Fix policy learning (traced failures → D-021…D-026): cumulative target chunks, minimal object-relative observation, executor ready pose, goal relabelling of pseudo-demos.
2. [x] M4 core experiment: video vs teleop at 1×/2×/4× human time, learning curve, ablations (−retargeting, −object-centric, +sim pre-training, +self-practice), 3 seeds × 80 trials.
3. [x] M3 metrics + sensitivity sweeps; M5 active requests (3 strategies × 5 rounds × 3 seeds); M6 detector + recovery with blind-retry and oracle baselines; M7 chores 2–4, held-out chore, generalization matrix; safety audit of the measured state.
4. [x] Real-robot path: `eval/session_protocols.py`, `hardware/real_env.py`, `RayPlanePerceiver`, camera→table extrinsics, hue calibration, `eval/queue.py`, SESSION_QUEUE S1–S6, HUMAN_TODO H1–H9.
5. [x] M8 teaching app + scripted sim walkthrough.
6. [x] Claim pipeline: `bench/numbers.py` → `bench/make_report.py` (templates in `docs/templates/`) → `bench/check_claims.py`; `bench/repo_scan.py`; `make reproduce-quick` from a clean clone.

## Single next action
**Human:** buy and assemble the arm (HUMAN_TODO H1–H3), then run S1 (safety smoke). After that, film 10 desk-clearing clips (H5) so the agent can bring up the real-video reader and train the first real policy for S2.
