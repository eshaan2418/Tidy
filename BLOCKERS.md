# BLOCKERS

- **B-001 (2026-10-24) — No robot arm.** Fallback: simulation only (D-001). Impact: no `real` results. M1 real bring-up, the M2 teleop baseline, and all real evaluation are `awaiting-human` with scripts in HUMAN_TODO.md / SESSION_QUEUE.md.
- **B-002 (2026-10-24) — No NVIDIA GPU (Apple M5 Pro, MPS).** Fallback: small ACT / Diffusion Policy on low-dim observations. Impact: the `−VLA pretraining` ablation and VLA fine-tuning (SmolVLA/OpenVLA/π0-class) are not run. They are listed as future work, not as results.
- **B-003 (2026-10-24) — No phone videos yet.** Fallback: synthetic phone videos rendered in the digital twin with a human-hand proxy and coloured fingertip caps. Hand keypoints come from colour blobs plus depth, a stand-in for a HaMeR/WiLoR-class estimator. Impact: M3 perception metrics are `sim`. Real-video perception accuracy is unmeasured until the filming session (HUMAN_TODO H5).
