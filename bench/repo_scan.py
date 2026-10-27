"""Definition-of-Done check 5: no credentials, raw private video, or large files anywhere in git history.

Scans every blob reachable from any ref: path rules (data/private/, raw video/image extensions outside the allowed
result-figure folders), size (> 5 MB), and secret patterns (API keys, private keys, tokens). Exit 1 on findings.
"""

from __future__ import annotations

import re
import subprocess
import sys

from common.provenance import REPO_ROOT

MAX_BYTES = 5 * 1024 * 1024
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".heic", ".m4v", ".r3d", ".exr")
SECRET_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "aws access key"),
    (r"-----BEGIN (RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY-----", "private key"),
    (r"ghp_[A-Za-z0-9]{36}", "github token"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "slack token"),
    (r"sk-[A-Za-z0-9]{32,}", "api key (sk-)"),
    (r"hf_[A-Za-z0-9]{30,}", "hugging face token"),
    (r"(?i)(password|passwd|secret_key)\s*[:=]\s*['\"][^'\"]{6,}['\"]", "hard-coded password"),
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, errors="replace")


def scan() -> list[str]:
    findings: list[str] = []
    objs = git("rev-list", "--objects", "--all").splitlines()
    paths = {}
    for line in objs:
        parts = line.split(" ", 1)
        if len(parts) == 2:
            paths[parts[0]] = parts[1]
    if not paths:
        return findings
    batch = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                           input="\n".join(paths), cwd=REPO_ROOT, capture_output=True, text=True).stdout
    for line in batch.splitlines():
        sha, typ, size = line.split()
        if typ != "blob":
            continue
        p = paths[sha]
        if p.startswith("data/private/"):
            findings.append(f"private data in history: {p}")
        if p.lower().endswith(VIDEO_EXT):
            findings.append(f"video file in history: {p}")
        if int(size) > MAX_BYTES:
            findings.append(f"large file ({int(size) / 1e6:.1f} MB): {p}")
        if int(size) < 2_000_000 and not p.endswith((".png", ".pdf", ".pt", ".pkl", ".npz", ".jpg")):
            text = subprocess.run(["git", "cat-file", "-p", sha], cwd=REPO_ROOT, capture_output=True).stdout.decode(
                "utf-8", errors="ignore")
            for pat, what in SECRET_PATTERNS:
                if re.search(pat, text):
                    findings.append(f"possible {what}: {p}")
    return sorted(set(findings))


def main() -> int:
    f = scan()
    for x in f:
        print("FINDING:", x)
    print(f"repo scan: {len(f)} finding(s)")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
