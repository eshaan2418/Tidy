"""Release artefacts (M9): LeRobot-style datasets + privacy-safe processed human-demo manifests (sim stage).

  data/robot/sim_<chore>_video10/   retargeted pseudo-demos from 10 synthetic phone videos (seed 0)
  data/robot/sim_<chore>_teleop1x/  simulated teleop demos at equal human time (seed 0)
  data/human/<chore>/manifest.json  per clip: placement card, accept/reject, detected events, retargeted waypoints
No raw video is exported (the sim videos are synthetic; real clips stay in data/private/).
"""

from __future__ import annotations

import json

from bench.common import ledger_from_records, program_from_records, segments_from_records, teleop_data, video_records
from common.provenance import REPO_ROOT
from eval.chores import CHORES
from eval.placements import training_cards
from policies.data import dataset_hash, export_lerobot


def main() -> None:
    for chore in CHORES:
        recs = video_records(chore, training_cards(chore, "film", 10))
        prog = program_from_records(chore, recs)
        segs = segments_from_records(recs, "video", prog)
        export_lerobot(segs, f"sim_{chore}_video10", extra_meta={"stage": "sim", "source": "synthetic phone video",
                                                                 "chore": chore})
        tsegs, L, _, _ = teleop_data(chore, ledger_from_records(recs).total_s(), seed=0)
        export_lerobot(tsegs, f"sim_{chore}_teleop1x", extra_meta={"stage": "sim", "source": "simulated teleop",
                                                                   "chore": chore, "human_min": L.summary()})
        man = []
        for r in recs:
            man.append({"card": r.card, "accepted": r.accepted, "reject_reasons": r.reject_reasons,
                        "video_duration_s": r.video_duration_s,
                        "picks": [{"obj": s.obj, "accepted": s.accepted, "feasible_frac": s.feasible_frac,
                                   "pick_pos": s.pick_pos.tolist(), "place_pos": s.place_pos.tolist(),
                                   "waypoints": [[*w.pos.tolist(), w.yaw, w.pitch, w.grip] for w in s.waypoints]}
                                  for s in r.segments]})
        out = REPO_ROOT / "data" / "human" / chore
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({"stage": "sim", "dataset_hash": dataset_hash(segs),
                                                       "program": {k: [v.landmark, list(v.offset)]
                                                                   for k, v in prog.program.items()},
                                                       "clips": man}))
        print(chore, len(segs), "video segments,", len(tsegs), "teleop segments")


if __name__ == "__main__":
    main()
