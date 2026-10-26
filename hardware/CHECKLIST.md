# Hardware bring-up checklist (M1)

Do these steps in order. Tick each box, write the measured values here, and commit the file.

## A. Assembly (~2–3 h)
- [ ] Follow the official SO-101/SO-100 assembly guide (github.com/TheRobotStudio/SO-ARM100 and the LeRobot docs "SO-101" page) for the follower arm, and for the leader arm if you have it.
- [ ] Set servo IDs 1–6 with the LeRobot motor-setup command for your LeRobot version, **one servo at a time**.
- [ ] Route cables so that no cable is pinched at full range of motion. Move each joint by hand with the power off.
- [ ] Clamp the base to the table, with the base centre ~8 cm from the near edge of the mat (the mat's row A is ~10 cm from the base axis).

## B. LeRobot calibration (~20 min)
- [ ] Run the LeRobot calibration command for the follower (and leader). Keep the generated calibration file.
- [ ] Power on and torque on, with **no** policy running and your hand on the e-stop. Check that each joint holds position.

## C. Tidy joint calibration → `hardware/calibration.yaml` (~20 min)
Run `python -m hardware.calibrate_joints --port <port>` (torque OFF). It walks you through each joint:
1. Move the arm by hand to the **zero pose** (arm straight up, gripper jaw axis parallel to the table's left-right axis) and press Enter. This records `offset_deg` per joint.
2. Move each joint a few degrees in its positive direction (pan → left, lift/elbow/wrist → forward/down, roll → clockwise seen from the gripper) and press Enter. This records `sign`.
3. Open the gripper fully, then close it fully. This records `open` and `closed`.
- [ ] Measured link lengths (mm, axis to axis): upper arm ____, forearm ____, wrist pivot → fingertip centre ____. If any differs by more than 3 mm from `common/kinematics.py`, tell the agent.

## D. Cameras (~20 min)
- [ ] Mount the overhead camera ~60 cm above the mat centre, looking straight down. Mount the side camera at ~45°.
- [ ] Camera intrinsics: run `python -m hardware.camera_calib intrinsics --cam 0` and show the printed ChArUco board in ~20 poses. Target RMS reprojection error < 0.5 px.
- [ ] Extrinsics: lay the ChArUco board flat on the mat at the marked position and run `python -m hardware.camera_calib extrinsics --cam 0`.
- [ ] Repeat both steps for the side camera.

## E. Latency + repeatability (~20 min, supervised)
- [ ] Measure camera latency: `python -m hardware.latency camera --cam 0` (point it at the laptop's on-screen flashing timer).
- [ ] Measure control-loop latency: this is reported by the safety-smoke session (S1) log as the tick period and the e-stop stop time.
- [ ] Repeatability: run the queued session R1 (`supervised_session.py eval --session R1`). The arm visits the same pose 50× and the overhead camera measures the spread of an ArUco marker on the gripper. Record: spread (mm, 95th percentile) ____.

## F. Safety review
- [ ] Run the safety smoke session S1 (HUMAN_TODO H4) and confirm it passes.
- [ ] Read SAFETY_LOG.md and confirm every entry is resolved.
