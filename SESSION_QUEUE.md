# SESSION QUEUE (real robot)

Run only via `scripts/supervised_session.py`, with the operator present the whole time. Sessions are listed in the order they must be run.

| # | Session | Chore | Checkpoint | Placements | Trials | Command | Status |
|---|---|---|---|---|---|---|---|
| S1 | Safety smoke (no policy) | — | — | — | e-stop ×5, limit probes | `python scripts/supervised_session.py safety-smoke` | awaiting-human (needs arm) |
