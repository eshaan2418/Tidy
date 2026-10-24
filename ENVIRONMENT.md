# ENVIRONMENT

Probed 2026-10-24 (M0). Re-probe at every hardware change.

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M5 Pro (18 CPU cores, 20-core GPU, Metal 4) |
| RAM / disk | 48 GB unified / 745 GB free |
| OS | macOS 27.0.1 (26A434) |
| NVIDIA GPU / CUDA | **none** (PyTorch MPS available) |
| Python | 3.12.14 (mise), venv `.venv` via uv 0.12.13 |
| Key packages | mujoco 3.16.0, torch 2.14.1, numpy 2.5.4, opencv 5.0.0, scipy 1.18.1 (pinned in `requirements.lock`) |
| Docker | 29.7.2 server reachable |
| Robot arm | **not connected**: no USB serial device (only `/dev/cu.Bluetooth-Incoming-Port`, `/dev/cu.debug-console`) |
| Arm model / firmware / calibration | n/a: awaiting purchase/assembly (HUMAN_TODO H1–H3) |
| Cameras | MacBook Pro Camera (built-in); iPhone Continuity Camera visible. No USB webcams yet. |
| Phone depth | unknown: to be filled in when filming (iPhone Pro models have LiDAR) |
| LeRobot | not installed (no arm yet; driver adapter in `hardware/so100_driver.py` imports it lazily) |
| Sim stack | MuJoCo 3.16 offscreen rendering (CGL), custom SO-100-class MJCF in `sim/scene.py` |

## Fallback rungs chosen (see DECISIONS.md D-001)
- Robot: **Fallback 2, simulation only** (SO-100-class model). All real-robot claims are `awaiting-human`.
- Compute: **between Fallback 1 and 2**. Apple MPS/CPU only. We use ACT and Diffusion Policy at small scale with low-dimensional object-centric observations. VLA fine-tuning is blocked (BLOCKERS B-002).
- Human video: no phone videos yet. Sim stage uses **synthetic phone videos** rendered in the digital twin, with a human-hand proxy. The real pipeline targets phone RGB-D.
- Pretrained VLA: not used (compute). The `−VLA pretraining` ablation is reported as not run.
- Sim: MuJoCo scene approximating the planned real workspace.
