# PROGRESS

Evidence stage: **sim only** (no arm, no GPU, no phone videos yet; see BLOCKERS.md).

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap + prior work | in-progress | `docs/PRIOR_WORK.md` (108 verified citations), ENVIRONMENT.md |
| M1 Hardware + safety layer | in-progress (sim); real = awaiting-human | `safety/`, `tests/test_safety.py` |
| M2 Eval protocol + teleop baseline | todo | |
| M3 Human video pipeline | todo | |
| M4 Retargeting + policy learning | todo | |
| M5 Uncertainty-driven requests | todo | |
| M6 Failure detection + recovery | todo | |
| M7 Multi-chore + generalization | todo | |
| M8 Teaching app | todo | |
| M9 Integration + write-up | todo | |

## Current plan (M0/M1)
1. [x] Env probe, fallback rungs (D-001)
2. [x] Kinematics (closed-form FK/IK), MJCF scene, sim env, scripted controller. 4/4 objects into the bin in a smoke run.
3. [x] Safety layer (`safety/core.py`, `safety/limits.py`)
4. [ ] Safety tests: limits can't be exceeded (randomized), e-stop latches, watchdog, proximity, driver-token bypass, e-stop present in every control loop (AST check)
5. [ ] Makefile, CI workflow, lockfile, Dockerfile; ruff + mypy clean
6. [ ] HUMAN_TODO: shopping list, assembly/calibration checklist, filming guide
7. [ ] supervised_session.py (sim mode testable; real mode gated)

## Single next action
Write `tests/test_safety.py` and `tests/test_kinematics_env.py`, then wire up the Makefile and CI.
