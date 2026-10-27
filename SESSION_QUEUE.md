# SESSION QUEUE (real robot)

Run only via `scripts/supervised_session.py`, with the operator present the whole time. Sessions are listed in the order they must be run. Each eval session is created with `python -m eval.queue --chore <chore> --session <id> --conditions ...`. Re-create it after the real-data policies exist, because the checkpoints must be the ones trained on **real** videos / teleop demos. Trial order is randomised across methods within a session, and every method gets the same placement cards.

| # | Session | Chore | Checkpoints | Placements | Trials | Command | Expected | Status |
|---|---|---|---|---|---|---|---|---|
| S1 | Safety smoke (no policy) | — | — | — | 4 slow moves, 5 e-stop drills, limit probes | `python scripts/supervised_session.py safety-smoke` | 15 min | awaiting-human (needs arm, H1–H3) |
| T1 | Teleop demo collection (equal human time) | desk_clearing | — | `teleop` cards | until budget | HUMAN_TODO H7 (LeRobot record) | ≈ filming time | awaiting-human (needs leader arm) |
| S2 | Video vs teleop (core question) | desk_clearing | `video_10` (real videos) vs `teleop_1x` (real teleop) | eval_seen #0–19 + eval_unseen_place #0–19 | 40 per method = 80 | `python -m eval.queue --session S2 --conditions video_10 teleop_1x` then `python scripts/supervised_session.py eval --session S2` | ≈ 2 h (split into 2 sittings) | awaiting-human (needs H5, H7) |
| P1 | Self-practice round | desk_clearing | `video_10` | `practice` cards #0–19 | 20 episodes | `python scripts/supervised_session.py practice --session P1` | 30 min | awaiting-human |
| S3 | Active requests (after 5 rounds of 2 requested videos per strategy) | desk_clearing | `uncertainty`, `random`, `round_robin` (real) | eval_seen #0–19 | 20 per strategy = 60 | `eval.queue --session S3 --conditions uncertainty random round_robin` | 1.5 h | awaiting-human |
| S4 | Recovery on/off | desk_clearing | `video_10` ± recovery hook | eval_seen #20–39 | 20 + 20 | `eval.queue --session S4 --conditions video_10` (recovery flag in the session file) | 1 h | awaiting-human |
| S5 | Chores 2–4 + held-out | sorting, shelf_putaway, stacking, tray_heldout | `video_10` per chore | eval_seen #0–19 | 20 per chore | one `eval.queue` session per chore | 4 × 40 min | awaiting-human |
| S6 | Towel fold (stretch, deformable) | towel | `video_10` (towel) | 10 towel layouts | 10 | after filming 10 towel videos | 30 min | awaiting-human (not simulated, see BLOCKERS B-004) |

A dry run of the session protocol in the twin is part of `make reproduce-quick` (`eval.queue --session DRYRUN` + `supervised_session.py eval --sim --session DRYRUN`).
