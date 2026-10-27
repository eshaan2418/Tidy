"""M3 (sim stage): human-video processing accuracy against simulator ground truth (stands in for the
hand-labelled subset; real-video labelling instructions are in HUMAN_TODO.md H6).

Metrics per chore over N synthetic phone clips (film cards): grasp/release event precision/recall/F1 (0.3 s
tolerance), phase-boundary (event time) error, object pick/place position error (cm) vs ground truth, 3D hand
keypoint median error, clip acceptance rate, and moved-object identification.  Sweeps on desk clearing: simulated
hand-estimator noise, fingertip-cap colour detection instead of the estimator, and RGB-only (no phone depth).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bench.common import cmdline, video_records
from common.provenance import write_result
from eval.chores import CHORES
from eval.placements import training_cards
from perception.pipeline import HandEstimatorNoise


def aggregate(recs) -> dict:
    tp = sum(r.metrics["event"]["tp"] for r in recs)
    fp = sum(r.metrics["event"]["fp"] for r in recs)
    fn = sum(r.metrics["event"]["fn"] for r in recs)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    be = np.array([e for r in recs for e in r.metrics["boundary_err_s"]])
    pick = np.array([e for r in recs for e in r.metrics["pick_err_m"]]) * 100
    place = np.array([e for r in recs for e in r.metrics["place_err_m"]]) * 100
    kp = {k: [r.metrics["keypoint_median_err_m"][k] for r in recs if r.metrics["keypoint_median_err_m"][k]]
          for k in ("thumb", "index")}

    def q(x):
        return {"median": float(np.median(x)), "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
                "p90": float(np.percentile(x, 90)), "n": int(len(x))} if len(x) else None

    return {"n_clips": len(recs), "accepted": int(sum(r.accepted for r in recs)),
            "accept_rate": float(np.mean([r.accepted for r in recs])),
            "reject_reasons": sorted({w for r in recs for w in r.reject_reasons}),
            "event": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / max(1e-9, prec + rec)},
            "boundary_err_s": q(be), "pick_err_cm": q(pick), "place_err_cm": q(place),
            "keypoint_err_cm": {k: (float(np.median(v)) * 100 if v else None) for k, v in kp.items()},
            "moved_objects_correct": float(np.mean([r.metrics["moved_objects_correct"] for r in recs])),
            "video_s_mean": float(np.mean([r.video_duration_s for r in recs]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 4 if args.quick else args.n
    t0 = time.time()
    chores = ["desk_clearing"] if args.quick else [c for c in CHORES]
    per_chore = {}
    for chore in chores:
        recs = video_records(chore, training_cards(chore, "film", n))
        per_chore[chore] = aggregate(recs)
        print(chore, per_chore[chore]["event"]["f1"], per_chore[chore]["pick_err_cm"], flush=True)
    sweeps = {}
    if not args.quick:
        cards = training_cards("desk_clearing", "film", 12)
        for sxy in (0.003, 0.006, 0.012, 0.024):
            nz = HandEstimatorNoise(sigma_xy=sxy, sigma_z=2 * sxy)
            sweeps[f"hand_noise_{sxy * 1000:.0f}mm"] = aggregate(video_records("desk_clearing", cards, noise=nz))
        sweeps["fingertip_caps"] = aggregate(video_records("desk_clearing", cards, hand_source="caps"))
        sweeps["rgb_only"] = aggregate(video_records("desk_clearing", cards, rgb_only=True))
        for k, v in sweeps.items():
            print(k, v["event"]["f1"], v["accept_rate"], v["pick_err_cm"]["median"] if v["pick_err_cm"] else None)
    out = write_result("sim/m3_perception" + ("_quick" if args.quick else ""),
                       {"per_chore": per_chore, "sweeps": sweeps, "wall_s": time.time() - t0}, stage="sim",
                       config={"n_clips": n, "event_tolerance_s": 0.3, "default_noise": HandEstimatorNoise().__dict__},
                       seed=0, command=cmdline())
    print("wrote", out)


if __name__ == "__main__":
    main()
