"""ChArUco camera calibration (intrinsics + table extrinsics). Human-operated with a printed board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "cameras"
SQUARES, SQUARE_M, MARKER_M = (7, 5), 0.030, 0.022


def board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard(SQUARES, SQUARE_M, MARKER_M, d)


def write_board_png(path: Path) -> None:
    img = board().generateImage((7 * 300, 5 * 300))
    cv2.imwrite(str(path), img)


def detect(gray, b):
    det = cv2.aruco.CharucoDetector(b)
    ch_corners, ch_ids, _, _ = det.detectBoard(gray)
    return ch_corners, ch_ids


def intrinsics(cam: int, n: int = 20) -> None:
    b = board()
    cap = cv2.VideoCapture(cam)
    objp: list = []
    imgp: list = []
    size: Any = None
    while len(objp) < n:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        c, ids = detect(gray, b)
        cv2.imshow("calib (space=capture, q=quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and ids is not None and len(ids) > 8:
            o, i = b.matchImagePoints(c, ids)
            objp.append(o)
            imgp.append(i)
            print(f"captured {len(objp)}/{n}")
    rms, K, dist, _, _ = cv2.calibrateCamera(objp, imgp, size, None, None)
    OUT.mkdir(exist_ok=True)
    (OUT / f"cam{cam}_intrinsics.json").write_text(json.dumps({"rms": rms, "K": K.tolist(),
                                                               "dist": dist.ravel().tolist(), "size": size}))
    print("RMS reprojection error:", rms, "(target < 0.5 px)")


def extrinsics(cam: int) -> None:
    intr = json.loads((OUT / f"cam{cam}_intrinsics.json").read_text())
    K, dist = np.array(intr["K"]), np.array(intr["dist"])
    b = board()
    cap = cv2.VideoCapture(cam)
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c, ids = detect(gray, b)
    o, i = b.matchImagePoints(c, ids)
    ok, rvec, tvec = cv2.solvePnP(o, i, K, dist)
    (OUT / f"cam{cam}_extrinsics.json").write_text(json.dumps({"rvec": rvec.ravel().tolist(),
                                                               "tvec": tvec.ravel().tolist(),
                                                               "board_origin_in_table_frame": BOARD_ORIGIN.tolist()}))
    print("saved extrinsics (board frame; board origin placed at mat corner A1 per CHECKLIST D)")


# Board laid face-up on the mat, origin at the corner of mat cell A1 nearest the robot's right
# (BOARD_ORIGIN in the table frame), board x-axis pointing away from the robot. OpenCV's board frame has z into the
# board, so table <- board is a rotation by pi about x.
BOARD_ORIGIN = np.array([0.10, -0.105, 0.0])
R_TABLE_BOARD = np.diag([1.0, -1.0, -1.0])


def table_extrinsics(rvec: np.ndarray, tvec: np.ndarray, origin: np.ndarray = BOARD_ORIGIN):
    """Board-frame PnP pose -> (R, t) mapping TABLE-frame points into the camera frame (OpenCV convention)."""
    R_cb, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    R = R_cb @ R_TABLE_BOARD.T
    t = np.asarray(tvec, float).ravel() - R @ origin
    return R, t


def overhead(cam: int) -> None:
    """Write hardware/cameras/overhead.json (K, R, t in the table frame) used by hardware/real_env.py."""
    intr = json.loads((OUT / f"cam{cam}_intrinsics.json").read_text())
    ext = json.loads((OUT / f"cam{cam}_extrinsics.json").read_text())
    R, t = table_extrinsics(np.array(ext["rvec"]), np.array(ext["tvec"]))
    (OUT / "overhead.json").write_text(json.dumps({"K": intr["K"], "dist": intr["dist"], "R": R.tolist(),
                                                   "t": t.tolist(), "cam": cam}))
    print("saved hardware/cameras/overhead.json")


def hues(cam: int) -> None:
    """Session-start colour calibration (D-018): click once on each object, type its colour class."""
    from sim.scene import HUES

    cap = cv2.VideoCapture(cam)
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    picks: list[tuple[int, int]] = []
    cv2.imshow("click each object once, then press q", frame)
    cv2.setMouseCallback("click each object once, then press q", lambda e, x, y, *_: picks.append((x, y))
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    while (cv2.waitKey(20) & 0xFF) != ord("q"):
        pass
    out = {}
    for x, y in picks:
        name = input(f"colour class at ({x},{y}) {sorted(HUES)}: ").strip()
        patch = hsv[max(0, y - 3):y + 4, max(0, x - 3):x + 4, 0].astype(float) * 2
        a = np.deg2rad(patch.ravel())
        out[name] = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)
    (OUT / "hues.json").write_text(json.dumps(out, indent=1))
    print("saved hardware/cameras/hues.json:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["intrinsics", "extrinsics", "overhead", "hues", "board"])
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "board":
        write_board_png(Path(__file__).resolve().parent / "charuco.png")
    elif a.mode == "intrinsics":
        intrinsics(a.cam)
    elif a.mode == "overhead":
        overhead(a.cam)
    elif a.mode == "hues":
        hues(a.cam)
    else:
        extrinsics(a.cam)


if __name__ == "__main__":
    main()
