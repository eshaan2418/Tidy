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

## H6. Label a small real-video subset for the M3 accuracy check (~40 min) — awaiting-human (after H5)
The sim-stage perception metrics use simulator ground truth. Real accuracy needs your labels on **6 desk_clearing clips**:
1. Open each clip in any player that shows the frame time (e.g., VLC, View → Time display in ms).
2. For every pick-place, write one row to `data/private/labels/desk_clearing.csv`: `clip,object,grasp_s,release_s,start_cell,end` where `grasp_s` is when the fingers first close on the object, `release_s` is when they open over the goal, and `end` is `bin` / the goal.
3. With a ruler, measure 3 resting objects per clip against the printed mat grid (cm from the nearest cell centre). This gives the object-position error ground truth.
4. Run `python -m bench.m3_perception_real` (it is written once the clips exist, since the reader depends on your phone's export format). It reports event F1, boundary error and position error, the same metrics as the sim table.

## H7. Teleop baseline demos at equal human time (~10–15 min) — awaiting-human (needs the leader arm)
1. Read your logged desk_clearing filming minutes in `data/private/human_time.csv`. That total (filming + setup + resets) is your **teleop budget**. Start a timer.
2. Record leader-arm demos with LeRobot's recording command for your LeRobot version (`lerobot-record` / `python -m lerobot.record`): robot = SO-101 follower, teleop = SO-101 leader, fps = 10, task = "put every item into the bin". Save to `data/robot/teleop_desk_real/`.
3. Before each demo, set up the next teleop placement card: `python -c "from eval.placements import training_cards as t; [print(i, c.human_text()) for i, c in enumerate(t('desk_clearing','teleop',30))]"`.
4. Stop as soon as the timer reaches the budget. **Every** minute counts, including resets, failed demos and re-takes.
5. Log the minutes, then ask the agent to convert the dataset (`policies/data.py` spec) and train `teleop_1x` with the identical config used for the video policy.

## H8. Evaluation sessions — awaiting-human
See `SESSION_QUEUE.md`. Each session lists its checkpoints, placement cards, trial count and expected duration. All sessions run through `python scripts/supervised_session.py eval --session <id>`. Before each session:
- recalibrate the overhead camera, ~3 min. Lay the ChArUco board flat with its origin at the A1 corner and its x-axis pointing away from the robot, then run `python -m hardware.camera_calib extrinsics --cam <idx>` followed by `python -m hardware.camera_calib overhead --cam <idx>`. Intrinsics are needed once only: `... intrinsics --cam <idx>`;
- click each object once in the hue calibration: `python -m hardware.camera_calib hues --cam <idx>`;
- review `SAFETY_LOG.md`. The script refuses to start while any entry is unresolved.

During the session, score each object honestly when asked (y/n). Afterwards, review 20 % of the trial videos (`data/private/sessions/<id>/`, chosen by the script) and correct any score you got wrong.

## H9. Teach the held-out chore with the app (M8, ~30 min) — awaiting-human
Follow **only** the app, starting with `python scripts/teach.py new tray_heldout`. It tells you every next command: guide → film → upload → process → train → try (a supervised session) → requests → record → try. The app logs your minutes. Report anything confusing, and the agent will fix the app.
