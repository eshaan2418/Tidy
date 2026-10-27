# Tidy — 2-minute demo script

Two versions: **sim** (runs today, no hardware) and **real** (needs the arm, H1–H4 in `HUMAN_TODO.md`). Both show the same story: *film → train → robot does the chore → robot asks for one demo → improves.* This file contains no result numbers. The numbers live in `README.md` and come from `results/`.

## Sim version (screen recording, ~2 min)

| Time | On screen | Command / action | Voice-over |
|---|---|---|---|
| 0:00–0:15 | Printed-mat render of the desk-clearing scene | `python -m hardware.make_mat` and open `hardware/mat.pdf` | "This is a $250-class arm's workspace. I want it to tidy my desk, and I don't want to teleoperate it." |
| 0:15–0:35 | Three synthetic phone clips playing | `python scripts/teach.py new tray_heldout`, then `guide`, then `upload --sim` | "I film myself doing the chore ten times on my phone. Each clip is about half a minute, and the app tells me where to put things." |
| 0:35–0:55 | Processing log: per-clip OK/REJECTED, then "What I learned" | `python scripts/teach.py process tray_heldout` | "It finds my hand, the grasps and releases, and where each object went, and it infers the task: pen and eraser onto the tray, cup stays." |
| 0:55–1:15 | Twin rollout video (overhead camera) | `python scripts/teach.py train tray_heldout`, then `try --sim` | "The robot practises in its digital twin. Some items work and some don't." |
| 1:15–1:35 | Key-frame PNG with the object circled, plus the request text | `python scripts/teach.py requests tray_heldout` | "Instead of asking for more of everything, it asks for the one demo it is least sure about: *'Place the eraser at B2, rotated 90°, and show me once more how you pick it up.'*" |
| 1:35–2:00 | Second try, then the status line with total human minutes | `record --sim`, `process`, `train`, `try --sim`, `status` | "One more clip and it's better. Every minute I spent is logged. That is the budget we compare against teleoperation." |

Shortcut: `make teach-demo` runs all of the above non-interactively and writes `results/sim/m8_teach_tray_heldout.json`.

## Real version (once hardware exists)

1. Film the desk-clearing clips (H5). Only hands should be in frame; faces are blurred if they appear.
2. `python scripts/teach.py upload desk_clearing <clips-dir>`, then `process` and `train`.
3. `python -m eval.queue --chore desk_clearing --session DEMO --conditions teach_v10 --n-seen 3 --n-unseen 0`, then `python scripts/supervised_session.py eval --session DEMO`. Film the robot with a second phone, with the operator's hand next to the e-stop in view.
4. `python scripts/teach.py requests desk_clearing`. Show the key frame, film the requested clip, retrain, and run the session again.
5. Cut to the trial videos from `data/private/sessions/DEMO/`. Show only processed, face-free footage.

**Safety on camera:** the operator is visible and in reach of the e-stop in every real shot. No unattended operation, no people or pets near the arm.
