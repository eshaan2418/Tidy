# HUMAN TODO

The agent cannot do these steps. Each item has an exact procedure and an expected duration. Status is `awaiting-human` unless marked otherwise.
**Log every minute you spend** in `data/private/human_time.csv` (`date,chore,activity,minutes`). This feeds the human-time accounting (Rule 5).

## H1. Buy hardware (~$250–450) — awaiting-human
- [ ] SO-101 (or SO-100) **follower** arm kit: 6× Feetech STS3215 servos (12 V version recommended), controller board, 12 V PSU, printed parts. Search "SO-101 arm kit" or see github.com/TheRobotStudio/SO-ARM100.
- [ ] SO-101 **leader** arm kit (needed only for the teleop baseline, M2)
- [ ] 2× USB webcams, 1080p/30 fps (one overhead on a stand ~60 cm above the mat, one side view). The overhead camera doubles as the proximity monitor.
- [ ] USB e-stop button that enumerates as a keyboard (or use the space bar; keep the laptop within arm's reach)
- [ ] 2× table clamps for the arm base
- [ ] Printed placement mat (the agent will generate `hardware/mat.pdf` in M2) + ChArUco board (`hardware/charuco.pdf`)
- [ ] Objects: pen, mug-sized plastic cup, tape roll, small cardboard box (~8×4×4 cm), a plastic bin (~11×9 cm opening, ≤5 cm wall), 4 coloured blocks (2 red, 2 blue), 3 plastic plates/cups of different colours, a small shelf/board (~13×10 cm, ~4.5 cm high), a washcloth. **No glass, knives, liquids, or hot items.**
- [ ] Optional: coloured fingertip caps or tape (lime on thumb, pink on index). These match the sim pipeline's fingertip detector and are the fallback if a learned hand-pose model underperforms.

## H2. Assemble + calibrate (~3–4 h) — awaiting-human
Follow `hardware/CHECKLIST.md` step by step. At the end, `hardware/calibration.yaml` must exist (the driver refuses to run without it).

## H3. Verify the LeRobot driver (~20 min, with the agent's help) — awaiting-human
`pip install lerobot`, then run `python -m hardware.check_driver --port <port>`. This command only *reads* joint positions; torque stays off. Paste the output into a new session. The agent then checks the adapter against the installed LeRobot API.

## H4. Safety smoke session S1 (~15 min) — awaiting-human
`python scripts/supervised_session.py safety-smoke --port <port> --monitor-cam 0 --scene-cam 1`
- The arm moves slowly between 4 high poses, then runs 5 e-stop drills (press the e-stop when prompted) and limit probes.
- Expected: `"pass": true`. Any surprise → write it in SAFETY_LOG.md and stop.

## H5. Film human demos (~25 min per chore) — awaiting-human
**Filming guide (all chores):**
1. Same table, mat taped down, the robot parked (powered off) at its home pose, lights on, and no windows behind the table.
2. Phone held **landscape**, standing on the human side of the table (opposite the robot), about 50–70 cm from the mat and 40–50 cm above it, angled down ~35°. Keep **all four mat corner markers visible the whole time**. If your phone has LiDAR (iPhone Pro), record with depth using an app that saves RGB-D (e.g., Record3D). Otherwise record normal video at 30 fps.
3. Before each clip, place the objects using the placement card the agent generates (`data/human/<chore>/placements_film.json`, printed as grid cells such as "cup at C4, handle-left").
4. Press record, wait 1 s with your hand out of view, do the chore at a **natural but deliberate** pace with **one hand**, pinch each object between thumb and index finger, then remove your hand and wait 1 s.
5. Faces must not be in frame. If they are, the clip is blurred before any use outside `data/private/`.
6. Save clips as `data/private/videos/<chore>/clip_XX.mov`. The clip count per chore is 10, and each clip uses its own placement card.
7. Log the minutes spent on filming, resets, and re-takes.

Per-chore instructions:
- **desk_clearing**: put every item (pen, cup, tape, box) into the bin, in any order.
- **sorting**: red blocks into the left bin, blue blocks into the right bin.
- **shelf_putaway**: put the 3 items upright onto the shelf.
- **stacking**: leave the blue plate where it is. Stack red on blue, then yellow on red.
- **held-out**: filmed only during the M8 app test (see H8).

## H6. Teleop baseline demos (equal human time) — awaiting-human (needs leader arm)
Instructions are generated in M2 (`SESSION_QUEUE.md` session T1). The time budget equals your logged desk_clearing filming time.

## H7. Evaluation sessions — awaiting-human
See `SESSION_QUEUE.md`. Each session lists its checkpoint, placement cards, trial count, and expected duration.

## H8. Teach the held-out chore with the app (M8) — awaiting-human
`python -m scripts.tidy_app` → "New chore". Follow only the on-screen instructions and log your minutes.
