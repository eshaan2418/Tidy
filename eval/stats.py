"""Statistics: Wilson intervals, Newcombe difference CIs, Fisher exact tests, bootstrap."""

from __future__ import annotations

import numpy as np
from scipy import stats as st


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Returns (p_hat, lo, hi) Wilson score 95% interval."""
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(p), float(max(0.0, c - h)), float(min(1.0, c + h))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe (method 10) 95% CI from Wilson intervals."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(d), float(lo), float(hi)


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    return float(st.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_rate(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{100*p:.0f}% [{100*lo:.0f}, {100*hi:.0f}] (n={n})"
