"""Render a sim trial montage for the README: video_10 (seed 0) clearing the desk on eval_seen cards, overhead view.
Writes paper/figures/sim_rollout.gif. Clearly labelled `sim`; no person is ever in frame."""

from __future__ import annotations

import cv2
import imageio.v2 as imageio

from bench.common import program_from_records, video_records
from bench.m6_recovery import load_ckpts
from common.provenance import REPO_ROOT
from eval.chores import CHORES
from eval.placements import load_protocol, training_cards
from eval.runner import make_env, run_episode
from perception.overhead import OverheadPerceiver
from policies.agent import PolicyAgent


def main() -> None:
    chore = CHORES["desk_clearing"]
    prog = program_from_records(chore.name, video_records(chore.name, training_cards(chore.name, "film", 10)))
    pols = load_ckpts(chore.name, "video_10", 0)
    frames = []
    for card in load_protocol(chore.name, "eval_seen")[:3]:
        env = make_env(chore, card)
        orig = env.step

        def step(a, monitor_rgb=None, orig=orig, env=env, card=card):
            info = orig(a, monitor_rgb)
            if env.step_count % 2 == 0:
                f = cv2.resize(env.render("overhead", 320, 240), (240, 180))
                cv2.putText(f, f"sim | video_10 | card {card.seed % 1000}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (20, 20, 20), 1)
                frames.append(f)
            return info

        env.step = step  # type: ignore[method-assign]
        run_episode(chore, card, PolicyAgent(pols), OverheadPerceiver(), program=prog.program, order=prog.order,
                    env=env)
        env.close()
    out = REPO_ROOT / "paper" / "figures" / "sim_rollout.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames[::2], duration=0.1, loop=0)
    print(out, len(frames))


if __name__ == "__main__":
    main()
