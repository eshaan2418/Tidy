"""Printable placement mat (true scale) + ChArUco calibration board.

    python -m hardware.make_mat      # -> hardware/mat.pdf, hardware/charuco.pdf

Mat: table-frame grid rows A..E (A nearest the robot) x columns 1..6 (1 = robot's right), held-out cells shaded,
4 ArUco markers (DICT_4X4_50, ids 0-3) at the corner positions used by perception (sim/scene.MARKER_POS), and the
fixture footprints (bin / shelf / tray). Print at 100 % on A3 (or two A4 sheets taped); check the 10 cm scale bar.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.placements import ALL_CELLS, HELD_OUT_CELLS, cell_center  # noqa: E402
from sim.scene import FIXTURES, MARKER_HALF, MARKER_POS  # noqa: E402

OUT = Path(__file__).resolve().parent
INCH = 0.0254


def make_mat(path: Path = OUT / "mat.pdf") -> Path:
    # table frame: x away from robot, y to the robot's left. Page: robot at the bottom, x up, y to the LEFT.
    x0, x1, y0, y1 = -0.05, 0.34, -0.31, 0.31
    w, h = (y1 - y0) / INCH, (x1 - x0) / INCH
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(y1, y0)  # flip so +y is left on the page
    ax.set_ylim(x0, x1)
    ax.set_aspect("equal")
    ax.axis("off")
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i, (mx, my) in enumerate(MARKER_POS):
        img = cv2.aruco.generateImageMarker(d, i, 200)
        s = MARKER_HALF
        ax.imshow(img, cmap="gray", extent=(my + s, my - s, mx - s, mx + s), interpolation="nearest")
        ax.text(my, mx - s - 0.006, f"id {i}", ha="center", va="top", fontsize=7)

    for c in ALL_CELLS:
        cx, cy = cell_center(c)
        held = c in HELD_OUT_CELLS
        ax.add_patch(plt.Rectangle((cy - 0.024, cx - 0.018), 0.048, 0.036, fill=held, alpha=0.15 if held else 1,
                                   ec="0.5", fc="0.6", lw=0.5))
        ax.text(cy, cx, c, ha="center", va="center", fontsize=9, color="0.3")
        ax.plot([cy - 0.022, cy - 0.018], [cx - 0.016, cx - 0.016], color="0.6", lw=0.5)   # orientation tick
    spots: dict[tuple[float, float], list[str]] = {}
    for f in FIXTURES.values():
        ax.add_patch(plt.Rectangle((f.pos[1] - f.half[1], f.pos[0] - f.half[0]), 2 * f.half[1], 2 * f.half[0],
                                   fill=False, ec="0.3", ls="--", lw=0.6))
        spots.setdefault((round(f.pos[0], 3), round(f.pos[1], 3)), []).append(f.name)
    for (fx, fy), names in spots.items():
        ax.text(fy, fx, " / ".join(names), ha="center", va="center", fontsize=6, color="0.4")
    ax.plot([0.0], [0.0], marker="o", color="k")
    ax.text(0.0, 0.004, "robot base axis", ha="center", fontsize=7)
    ax.plot([y1 - 0.02, y1 - 0.02 - 0.10], [x1 - 0.03] * 2, color="k", lw=2)
    ax.text(y1 - 0.07, x1 - 0.026, "10 cm (check after printing)", ha="center", fontsize=7)
    ax.text(0.0, x1 - 0.01, "Tidy placement mat — shaded cells are held out (never used for filming/teleop)",
            ha="center", fontsize=8)

    fig.savefig(path)
    plt.close(fig)
    return path


def make_charuco(path: Path = OUT / "charuco.pdf") -> Path:
    from hardware.camera_calib import MARKER_M, SQUARE_M, SQUARES, board

    img = board().generateImage((SQUARES[0] * 400, SQUARES[1] * 400))
    w, h = SQUARES[0] * SQUARE_M / INCH, SQUARES[1] * SQUARE_M / INCH
    fig = plt.figure(figsize=(w + 1, h + 1))
    ax = fig.add_axes((0.5 / (w + 1), 0.5 / (h + 1), w / (w + 1), h / (h + 1)))
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.axis("off")
    fig.text(0.5, 0.02, f"ChArUco {SQUARES[0]}x{SQUARES[1]}, square {SQUARE_M * 1000:.0f} mm, marker "
             f"{MARKER_M * 1000:.0f} mm, DICT_4X4_50 — print at 100 %", ha="center", fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(make_mat(), make_charuco())

