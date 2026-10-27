"""Definition-of-Done check 4: zero unbacked numbers in README, paper, resume bullets and the final review.

1. Every generated doc is current: re-rendering its template from results/ reproduces the file byte for byte.
2. Templates contain no hand-typed result numbers: after removing {{placeholders}}, no percentages, CI brackets,
   k/n fractions or 2+-decimal values remain (design constants go through bench.numbers.CONST).
3. Evidence stage: each generated doc states the stage of its numbers, and any `real` result file lists a trial
   video for every trial and those videos exist.
"""

from __future__ import annotations

import json
import re
import sys

from bench.make_report import KEY, build, targets
from common.provenance import REPO_ROOT, RESULTS_DIR

RESULT_LIKE = [
    (re.compile(r"(?<![\w.])\d+(\.\d+)?\s?%"), "percentage"),
    (re.compile(r"\[\s*[−-]?\d+(\.\d+)?\s*[–,-]\s*[−+-]?\d+(\.\d+)?\s*\]"), "interval"),
    (re.compile(r"(?<![\w./-])\d+/\d+(?![\w./-])"), "k/n fraction"),
    (re.compile(r"(?<![\w.])\d\.\d{2,}(?![\w.])"), "decimal"),
]


def scan_template(text: str) -> list[str]:
    stripped = KEY.sub("§", text)
    stripped = re.sub(r"`[^`]*`", "", stripped)           # code spans (paths, commands) are not claims
    stripped = re.sub(r"\]\([^)]*\)", "]", stripped)      # link targets
    stripped = re.sub(r"<!--.*?-->", "", stripped, flags=re.S)
    out = []
    for i, line in enumerate(stripped.splitlines(), 1):
        for pat, what in RESULT_LIKE:
            for m in pat.finditer(line):
                out.append(f"line {i}: hand-typed {what} '{m.group(0)}'")
    return out


def check() -> list[str]:
    problems: list[str] = []
    rendered = build()
    for tpl, target in targets():
        disk = (REPO_ROOT / target)
        if not disk.exists() or disk.read_text() != rendered[target]:
            problems.append(f"{target} is stale: run `make paper` (bench.make_report)")
        problems += [f"{tpl.name} {p}" for p in scan_template(tpl.read_text())]
        if not re.search(r"\b(sim|real)\b", rendered[target]):
            problems.append(f"{target}: no evidence-stage label (sim/real)")
    for p in RESULTS_DIR.rglob("*.json"):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or d.get("stage") != "real":
            continue
        trials = d.get("results", {}).get("trials", [])
        vids = d.get("videos", [])
        if len(vids) < len(trials):
            problems.append(f"{p.name}: {len(trials)} real trials but {len(vids)} videos")
        for v in vids:
            if not (REPO_ROOT / v).exists() and not __import__("pathlib").Path(v).exists():
                problems.append(f"{p.name}: trial video missing: {v}")
    return problems


def main() -> int:
    probs = check()
    for x in probs:
        print("CLAIM CHECK:", x)
    print(f"claim check: {len(probs)} problem(s)")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
