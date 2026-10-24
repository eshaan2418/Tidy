"""Provenance for every result file (Rule 1: never fabricate results).

Every JSON written to results/ goes through `write_result`, which stamps git commit,
dirty flag, hashes of the policy checkpoint and dataset, seed, hardware, config,
evidence stage (sim|real) and timestamp.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
VALID_STAGES = ("sim", "real", "offline")


def git_commit() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--"], cwd=REPO_ROOT,
                                stderr=subprocess.DEVNULL) != 0
        return {"sha": sha.decode().strip(), "dirty": bool(dirty)}
    except Exception:
        return {"sha": "no-git", "dirty": True}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version(), "cpus": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    try:
        import mujoco

        info["mujoco"] = mujoco.__version__
    except Exception:
        pass
    return info


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable hash of a JSON-able object or numpy arrays (dataset hash)."""
    h = hashlib.sha256()

    def _feed(o: Any) -> None:
        if isinstance(o, np.ndarray):
            h.update(str(o.dtype).encode())
            h.update(str(o.shape).encode())
            h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, dict):
            for k in sorted(o):
                h.update(str(k).encode())
                _feed(o[k])
        elif isinstance(o, (list, tuple)):
            for x in o:
                _feed(x)
        else:
            h.update(json.dumps(o, sort_keys=True, default=str).encode())

    _feed(obj)
    return h.hexdigest()


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def write_result(name: str, payload: dict[str, Any], *, stage: str, config: dict[str, Any], seed: int | None,
                 checkpoint_hash: str | None = None, dataset_hash: str | None = None,
                 videos: list[str] | None = None, command: str | None = None) -> Path:
    """Write results/<name>.json with full provenance. `stage` must be sim|real|offline."""
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}")
    if stage == "real" and not videos:
        raise ValueError("real-robot results require a video file per trial (Rule 1)")
    out = RESULTS_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "stage": stage,
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        "git": git_commit(),
        "hardware": hardware_info(),
        "seed": seed,
        "checkpoint_hash": checkpoint_hash,
        "dataset_hash": dataset_hash,
        "config": _jsonable(config),
        "command": command,
        "videos": videos or [],
        "results": _jsonable(payload),
    }
    out.write_text(json.dumps(record, indent=2, sort_keys=False))
    return out


def load_result(name: str) -> dict[str, Any]:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())
