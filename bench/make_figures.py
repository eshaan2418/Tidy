"""Regenerate every paper figure from results/sim/*.json (full runs). Missing inputs are skipped with a note.

    python -m bench.make_figures      # -> paper/figures/*.png

Palette: validated categorical slots (dataviz reference palette, light mode: all checks pass; contrast < 3:1 for
slots 3-4, so every bar carries a direct value label and every figure has a table in paper/paper.md).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.provenance import REPO_ROOT, RESULTS_DIR  # noqa: E402

FIG = REPO_ROOT / "paper" / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load(name: str) -> dict | None:
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _bars(ax, labels, groups: list[list[dict]], names: list[str]) -> None:
    n = len(groups)
    w = 0.8 / n
    x = np.arange(len(labels))
    for g, (vals, nm) in enumerate(zip(groups, names)):
        p = np.array([v["p"] for v in vals]) * 100
        lo = p - np.array([v["lo"] for v in vals]) * 100
        hi = np.array([v["hi"] for v in vals]) * 100 - p
        xs = x - 0.4 + w * (g + 0.5)
        ax.bar(xs, p, w - 0.02, color=C[g], label=nm, zorder=2)
        ax.errorbar(xs, p, yerr=[lo, hi], fmt="none", ecolor=INK2, lw=0.8, capsize=2, zorder=3)
        for xx, pp, top in zip(xs, p, p + hi):
            ax.text(xx, top + 1.0, f"{pp:.0f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylim(0, 108)


def fig_m4() -> str | None:
    r = _load("sim/m4_video_vs_teleop_desk_clearing")
    if r is None:
        return None
    s = r["summary"]
    order = [k for k in ("video_5", "video_10", "video_20", "teleop_1x", "teleop_2x", "teleop_4x", "teleop_fast_1x",
                         "video_raw", "video_hand", "video_simpre", "video_practice") if k in s]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURF)
    for ax, key, title in ((axes[0], "success", "Full-chore success"), (axes[1], "objects", "Items placed")):
        groups = [[s[k]["eval_seen"][key] for k in order], [s[k]["eval_unseen_place"][key] for k in order]]
        _bars(ax, order, groups, ["seen placements", "unseen placements"])
        _style(ax, f"{title} (%), 95% Wilson CI")
        ax.set_title(f"{title}: desk clearing (sim)", fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = FIG / "m4_video_vs_teleop.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    # human minutes vs success
    fig, ax = plt.subplots(figsize=(5, 3.4), facecolor=SURF)
    for i, (fam, keys) in enumerate((("phone video", ["video_5", "video_10", "video_20"]),
                                     ("teleop", ["teleop_1x", "teleop_2x", "teleop_4x"]))):
        keys = [k for k in keys if k in s]
        xm = [s[k]["eval_seen"]["success"]["human_min_mean"] for k in keys]
        allp = [(s[k]["eval_seen"]["success"]["k"] + s[k]["eval_unseen_place"]["success"]["k"]) /
                max(1, s[k]["eval_seen"]["success"]["n"] + s[k]["eval_unseen_place"]["success"]["n"]) for k in keys]
        ax.plot(xm, np.array(allp) * 100, "-o", color=C[i], lw=2, ms=8, label=fam)
        for xx, yy, k in zip(xm, allp, keys):
            ax.annotate(k, (xx, yy * 100), textcoords="offset points", xytext=(4, 5), fontsize=7, color=INK2)
    _style(ax, "Full-chore success (%), seen + unseen")
    ax.set_xlabel("Human teaching time (min)", color=INK2, fontsize=9)
    ax.set_title("Success vs human time (sim)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / "m4_human_time.png", dpi=180)
    plt.close(fig)
    return str(out)


def fig_m5() -> str | None:
    r = _load("sim/m5_active_desk_clearing")
    if r is None:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURF)
    for i, (st, pts) in enumerate(r["summary"].items()):
        x = np.array([p["n_videos"] for p in pts]) + (i - 1) * 0.12
        y = np.array([p["success_all"]["p"] for p in pts]) * 100
        lo = np.array([p["success_all"]["lo"] for p in pts]) * 100
        hi = np.array([p["success_all"]["hi"] for p in pts]) * 100
        ax.fill_between(x, lo, hi, color=C[i], alpha=0.12, lw=0)
        ax.plot(x, y, "-o", color=C[i], lw=2, ms=6, label=st.replace("_", "-"))
        ax.annotate(f"{y[-1]:.0f}", (x[-1], y[-1]), textcoords="offset points", xytext=(5, -3), fontsize=7, color=INK)
    _style(ax, "Full-chore success (%), 95% CI band")
    ax.set_xlabel("Phone videos (10 initial + requested)", color=INK2, fontsize=9)
    ax.set_title("Demo-request strategies: desk clearing (sim)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = FIG / "m5_active.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out)


def fig_m6() -> str | None:
    r = _load("sim/m6_recovery_desk_clearing")
    if r is None:
        return None
    modes = [m for m in ("no_recovery", "blind_retry_once", "recovery", "oracle_recovery") if m in r["pooled"]]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor=SURF)
    _bars(ax, [m.replace("_", " ") for m in modes], [[r["pooled"][m] for m in modes]], ["full-chore success"])
    _style(ax, "Full-chore success (%), 95% CI")
    ax.set_title("Failure detection + retry (sim)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    out = FIG / "m6_recovery.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out)


def fig_m7() -> str | None:
    ra, rb = _load("sim/m7_multichore_a"), _load("sim/m7_multichore_b")
    if ra is None and rb is None:
        return None
    r = {**(ra or {}), **(rb or {})}
    outs = []
    if "part_a" in r:
        t = r["part_a"]["table"]
        chores = sorted({k.split("/")[0] for k in t})
        fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=SURF)
        names = ["video_10", "teleop_1x", "video_10_multipre"]
        groups, used = [], []
        for nm in names:
            vals = []
            for ch in chores:
                d = t.get(f"{ch}/{nm}")
                vals.append(d["eval_seen"]["success"] if d else {"p": 0, "lo": 0, "hi": 0})
            if any(t.get(f"{ch}/{nm}") for ch in chores):
                groups.append(vals)
                used.append(nm)
        _bars(ax, chores, groups, used)
        _style(ax, "Full-chore success (%), seen placements")
        ax.set_title("Chores 2-4 + held-out chore (sim)", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "m7_chores.png", dpi=180)
        plt.close(fig)
        outs.append("m7_chores.png")
    if "part_b" in r:
        m = r["part_b"]["matrix"]
        tests = list(dict.fromkeys(k.split("/")[1] for k in m))
        fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=SURF)
        groups = [[m[f"{meth}/{tt}"]["objects"] for tt in tests] for meth in ("video_10", "teleop_1x")]
        _bars(ax, [tt.replace("_", " ") for tt in tests], groups, ["video_10", "teleop_1x"])
        _style(ax, "Items placed (%), 95% CI")
        ax.set_title("Generalization: desk clearing (sim)", fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "m7_generalization.png", dpi=180)
        plt.close(fig)
        outs.append("m7_generalization.png")
    return ", ".join(outs)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_m4, fig_m5, fig_m6, fig_m7):
        out = f()
        print(f.__name__, "->", out or "skipped (no full result yet)")


if __name__ == "__main__":
    main()
