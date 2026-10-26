"""Camera latency: film the laptop's own on-screen millisecond counter; latency = shown - captured time."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def camera(cam: int, n: int = 60) -> None:
    cap = cv2.VideoCapture(cam)
    lat = []
    for _ in range(n):
        t_show = time.monotonic()
        code = int(t_show * 1000) % 4096
        img = np.zeros((200, 600), np.uint8)
        for b in range(12):  # 12-bit binary code as white blocks
            if code >> b & 1:
                img[50:150, b * 50:(b + 1) * 50] = 255
        cv2.imshow("point the camera here", img)
        cv2.waitKey(1)
        ok, frame = cap.read()
        t_cap = time.monotonic()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (600, 200))
        bits = [(g[50:150, b * 50:(b + 1) * 50].mean() > 128) for b in range(12)]
        seen = sum(int(v) << b for b, v in enumerate(bits))
        d = ((int(t_cap * 1000) % 4096) - seen) % 4096
        lat.append(d)
    print(f"camera latency median {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms (n={len(lat)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["camera"])
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()
    camera(a.cam)
