"""make reproduce-quick | make reproduce-full.

quick (< 1 h on a laptop CPU, from a clean clone): safety tests, the sim safety smoke session, every experiment in
--quick mode (few clips / 1 seed / 6-10 trials per split) writing results/sim/*_quick.json, the M8 app walkthrough
on 4 trials, a session-protocol dry run, figure regeneration from the committed full results, the claim checker and
the repo scan. full: the complete experiment set (several hours on 16 cores), then figures + claim check.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from common.provenance import REPO_ROOT

PY = sys.executable


def run(args: list[str]) -> None:
    t = time.time()
    print("$", " ".join(args), flush=True)
    subprocess.run(args, cwd=REPO_ROOT, check=True)
    print(f"  ok ({time.time() - t:.0f} s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    run([PY, "-m", "pytest", "-q", "-m", "safety"])
    run([PY, "scripts/supervised_session.py", "safety-smoke", "--sim", "--session", "reproduce_smoke"])
    q = ["--quick"] if a.quick else []
    run([PY, "-m", "bench.m3_perception", *q])
    run([PY, "-m", "bench.m4_video_vs_teleop", *q])
    run([PY, "-m", "bench.m5_active", *q])
    run([PY, "-m", "bench.m6_recovery", *q])
    run([PY, "-m", "bench.m7_multichore", *q])
    run([PY, "scripts/teach.py", "walkthrough", "tray_heldout", "--sim", "--n", "4" if a.quick else "20"])
    run([PY, "-m", "eval.queue", "--chore", "desk_clearing", "--session", "DRYRUN", "--n-seen", "1", "--n-unseen", "1"])
    run([PY, "scripts/supervised_session.py", "eval", "--sim", "--session", "DRYRUN"])
    run([PY, "-m", "bench.make_figures"])
    run([PY, "-m", "bench.check_claims"])
    run([PY, "-m", "bench.repo_scan"])
    print(f"reproduce {'quick' if a.quick else 'full'}: all steps passed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
