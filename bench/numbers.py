"""Registry of every reportable number, computed from results/*.json (Rule 1). Docs reference them as {{key}}.

Each entry: key -> (text, source) where source = "results/<file>.json" (+ path hint). bench/make_report.py renders
docs/templates/*.md with these values; bench/check_claims.py verifies that rendered docs are current and that the
templates contain no hand-typed result numbers.
"""

from __future__ import annotations

import json

import numpy as np

from common.provenance import RESULTS_DIR
from eval.stats import newcombe_diff, wilson

# design constants (not results): written through the registry so the claim checker can tell them apart
CONST = {
    "const.n_videos": "10", "const.trials_seen": "40", "const.trials_unseen": "40", "const.seeds": "3",
    "const.reset_s": "20", "const.setup_s": "6", "const.request_overhead_s": "10", "const.k_requests": "2",
    "const.rounds": "5", "const.max_retries": "2", "const.audit_frac": "20%", "const.target": "50%",
    "const.budget_usd": "$250–450", "const.ctrl_hz": "10", "const.chunk": "10", "const.steps": "3000",
    "const.tcp_speed": "0.20", "const.pool": "30", "const.members": "3",
    "const.operator_speed": "60–90%", "const.ci": "95%", "const.orig_target": "80%",
}


def _load(name: str) -> dict | None:
    p = RESULTS_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def ci(d: dict) -> str:
    return f"{100 * d['p']:.0f}% [{100 * d['lo']:.0f}–{100 * d['hi']:.0f}]"


def frac(d: dict) -> str:
    return f"{d['k']}/{d['n']}"


def diff(a: dict, b: dict) -> str:
    d, lo, hi = newcombe_diff(a["k"], a["n"], b["k"], b["n"])
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(100 * d):.0f} pts [{100 * lo:+.0f}, {100 * hi:+.0f}]".replace("-", "−")


def both(s: dict, key: str = "success") -> dict:
    k = s["eval_seen"][key]["k"] + s["eval_unseen_place"][key]["k"]
    n = s["eval_seen"][key]["n"] + s["eval_unseen_place"][key]["n"]
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def registry() -> dict[str, tuple[str, str]]:
    R: dict[str, tuple[str, str]] = {k: (v, "constant") for k, v in CONST.items()}

    def put(key: str, text: str, src: str) -> None:
        R[key] = (text, src)

    # ---------------- M3
    src = "sim/m3_perception"
    r = _load(src)
    if r:
        for ch, m in r["results"]["per_chore"].items():
            put(f"m3.{ch}.f1", f"{m['event']['f1']:.2f}", src)
            put(f"m3.{ch}.prec", f"{m['event']['precision']:.2f}", src)
            put(f"m3.{ch}.rec", f"{m['event']['recall']:.2f}", src)
            put(f"m3.{ch}.bnd", f"{m['boundary_err_s']['median']:.2f} s" if m["boundary_err_s"] else "n/a", src)
            put(f"m3.{ch}.pick", f"{m['pick_err_cm']['median']:.1f} cm" if m["pick_err_cm"] else "n/a", src)
            put(f"m3.{ch}.pick90", f"{m['pick_err_cm']['p90']:.1f} cm" if m["pick_err_cm"] else "n/a", src)
            put(f"m3.{ch}.place", f"{m['place_err_cm']['median']:.1f} cm" if m["place_err_cm"] else "n/a", src)
            put(f"m3.{ch}.accept", f"{m['accepted']}/{m['n_clips']}", src)
            kp = m["keypoint_err_cm"]
            put(f"m3.{ch}.kp", f"{np.mean([v for v in kp.values() if v]):.1f} cm", src)
            put(f"m3.{ch}.moved", pct(m["moved_objects_correct"]), src)
            put(f"m3.{ch}.vid_s", f"{m['video_s_mean']:.0f} s", src)
        for k, m in r["results"]["sweeps"].items():
            put(f"m3.sweep.{k}.f1", f"{m['event']['f1']:.2f}", src)
            put(f"m3.sweep.{k}.accept", f"{m['accepted']}/{m['n_clips']}", src)
            put(f"m3.sweep.{k}.pick", f"{m['pick_err_cm']['median']:.1f} cm" if m["pick_err_cm"] else "n/a", src)
    # ---------------- M4
    src = "sim/m4_video_vs_teleop_desk_clearing"
    r = _load(src)
    if r:
        s = r["results"]["summary"]
        for c, d in s.items():
            put(f"m4.{c}.seen", ci(d["eval_seen"]["success"]), src)
            put(f"m4.{c}.unseen", ci(d["eval_unseen_place"]["success"]), src)
            put(f"m4.{c}.seen.obj", ci(d["eval_seen"]["objects"]), src)
            put(f"m4.{c}.unseen.obj", ci(d["eval_unseen_place"]["objects"]), src)
            put(f"m4.{c}.all", ci(both(d)), src)
            put(f"m4.{c}.all.frac", frac(both(d)), src)
            put(f"m4.{c}.all.obj", ci(both(d, "objects")), src)
            put(f"m4.{c}.min", f"{d['eval_seen']['success']['human_min_mean']:.1f}", src)
        if "video_10" in s and "teleop_1x" in s:
            put("m4.diff.video10_teleop1x", diff(both(s["video_10"]), both(s["teleop_1x"])), src)
        for a, b in (("video_10", "teleop_2x"), ("video_10", "teleop_4x"), ("video_10", "teleop_fast_1x"),
                     ("video_10", "video_raw"), ("video_10", "video_hand"), ("video_simpre", "video_10"),
                     ("video_practice", "video_10"), ("video_20", "video_10")):
            if a in s and b in s:
                put(f"m4.diff.{a}_{b}", diff(both(s[a]), both(s[b])), src)
        runs = r["results"]["runs"]
        for c in s:
            rs = [x for x in runs if x["name"] == c]
            n_demo = [x["meta"].get("n_demos") for x in rs if x["meta"].get("n_demos") is not None]
            if n_demo:
                put(f"m4.{c}.n_demos", f"{np.mean(n_demo):.1f}", src)
            nseg = [x["n_segments"] for x in rs]
            put(f"m4.{c}.n_segments", f"{np.mean(nseg):.0f}", src)
            pr = [x["meta"]["practice"] for x in rs if "practice" in x["meta"]]
            if pr:
                put(f"m4.{c}.practice_agree", pct(np.mean([p["auto_vs_true_agreement"] for p in pr])), src)
                put(f"m4.{c}.practice_kept", f"{np.mean([p['kept_steps'] for p in pr]):.0f}", src)
                put(f"m4.{c}.practice_steps", f"{np.mean([p['steps'] for p in pr]):.0f}", src)
            ops = [x["meta"].get("operator_success") for x in rs if x["meta"].get("operator_success") is not None]
            if ops:
                put(f"m4.{c}.operator_success", f"{sum(ops)}/{sum(n_demo)}", src)
        put("m4.n_trials_total", str(sum(len(sp["trials"]) for x in runs for sp in x["splits"].values())), src)
    src = "sim/m4_teleop_augmented_desk_clearing"
    r = _load(src)
    if r:
        for c, d in r["results"]["summary"].items():
            put(f"m4aug.{c}.all", ci(both(d)), src)
            put(f"m4aug.{c}.seen", ci(d["eval_seen"]["success"]), src)
            put(f"m4aug.{c}.unseen", ci(d["eval_unseen_place"]["success"]), src)
            put(f"m4aug.{c}.all.obj", ci(both(d, "objects")), src)
        m4 = _load("sim/m4_video_vs_teleop_desk_clearing")
        if m4 and "teleop_1x_aug" in r["results"]["summary"]:
            put("m4aug.diff.video10_teleop1xaug", diff(both(m4["results"]["summary"]["video_10"]),
                                                      both(r["results"]["summary"]["teleop_1x_aug"])), src)
    # ---------------- M5
    src = "sim/m5_active_desk_clearing"
    r = _load(src)
    if r:
        S = r["results"]["summary"]
        for st, pts in S.items():
            for p in pts:
                put(f"m5.{st}.r{p['round']}", ci(p["success_all"]), src)
                put(f"m5.{st}.r{p['round']}.min", f"{p['human_min']:.1f}", src)
            put(f"m5.{st}.final", ci(pts[-1]["success_all"]), src)
            put(f"m5.{st}.final.min", f"{pts[-1]['human_min']:.1f}", src)
            hit = [p["n_videos"] for p in pts if p["success_all"]["p"] >= 0.5]
            put(f"m5.{st}.to_target", f"{hit[0]} videos" if hit else "not reached", src)
            best = max(pts, key=lambda p: p["success_all"]["p"])
            put(f"m5.{st}.best", f"{ci(best['success_all'])} at {best['n_videos']} videos", src)
            k = sum(p["success_all"]["k"] for p in pts[1:])
            n = sum(p["success_all"]["n"] for p in pts[1:])
            pp, lo, hi = wilson(k, n)
            R[f"_m5.{st}.pooled"] = (json.dumps({"k": k, "n": n}), src)
            put(f"m5.{st}.pooled", ci({"p": pp, "lo": lo, "hi": hi}), src)
        if "uncertainty" in S and "random" in S:
            pa, pb = json.loads(R["_m5.uncertainty.pooled"][0]), json.loads(R["_m5.random.pooled"][0])
            d, lo, hi = newcombe_diff(pa["k"], pa["n"], pb["k"], pb["n"])
            if lo > 0:
                v, st_, sh = ("Uncertainty-driven requests beat random requests.", "Moderate (sim)", "supported")
            elif hi < 0:
                v, st_, sh = ("Random requests beat uncertainty-driven requests.", "Negative result (sim)",
                              "not supported")
            else:
                v, st_, sh = ("No significant difference between uncertainty-driven and random requests.",
                              "Inconclusive / negative (sim)", "not supported")
            put("m5.verdict", v, src)
            put("m5.strength", st_, src)
            put("m5.short", sh, src)
        for a, b in (("uncertainty", "random"), ("uncertainty", "round_robin")):
            if a in S and b in S:
                put(f"m5.diff.final.{a}_{b}", diff(S[a][-1]["success_all"], S[b][-1]["success_all"]), src)
                pa, pb = json.loads(R[f"_m5.{a}.pooled"][0]), json.loads(R[f"_m5.{b}.pooled"][0])
                put(f"m5.diff.pooled.{a}_{b}", diff(pa, pb), src)
        put("m5.n_requests", str(len(r["results"]["requests"])), src)
        ex = [q for q in r["results"]["requests"] if q["strategy"] == "uncertainty" and q["seed"] == 0]
        if ex:
            put("m5.example_request", ex[0]["text"], src)
            phases = [q["phase"] for q in r["results"]["requests"] if q["strategy"] == "uncertainty"]
            put("m5.unc_place_frac", pct(np.mean([p == "place" for p in phases])), src)
            objs = [q["obj"] for q in r["results"]["requests"] if q["strategy"] == "uncertainty"]
            top = max(set(objs), key=objs.count)
            put("m5.unc_top_obj", f"{top} ({objs.count(top)}/{len(objs)})", src)
    # ---------------- M6
    src = "sim/m6_recovery_desk_clearing"
    r = _load(src)
    if r:
        d = r["results"]["detector"]
        for part in ("val", "test"):
            put(f"m6.det.{part}.prec", f"{d[part]['precision']:.2f}", src)
            put(f"m6.det.{part}.rec", f"{d[part]['recall']:.2f}", src)
            put(f"m6.det.{part}.far", pct(d[part]["false_alarm_rate"]), src)
            put(f"m6.det.{part}.n", f"{d[part]['n_steps']}", src)
            pa = d[part]["phase_accuracy_on_true_positives"]
            put(f"m6.det.{part}.phase_acc", pct(pa) if pa is not None else "n/a", src)
        P = r["results"]["pooled"]
        for m, v in P.items():
            put(f"m6.{m}", ci(v), src)
            put(f"m6.{m}.frac", frac(v), src)
            put(f"m6.{m}.retries", f"{v['mean_retries']:.2f}", src)
            put(f"m6.{m}.safety", str(v["safety_events"]), src)
        put("m6.diff.recovery", diff(P["recovery"], P["no_recovery"]), src)
        put("m6.diff.oracle", diff(P["oracle_recovery"], P["no_recovery"]), src)
        if "blind_retry_once" in P:
            put("m6.diff.blind", diff(P["blind_retry_once"], P["no_recovery"]), src)
            put("m6.diff.recovery_vs_blind", diff(P["recovery"], P["blind_retry_once"]), src)
    # ---------------- M7
    src = "sim/m7_multichore"
    r = _load(src)
    if r:
        A = r["results"].get("part_a", {}).get("table", {})
        for k, d in A.items():
            key = k.replace("/", ".")
            put(f"m7.{key}.seen", ci(d["eval_seen"]["success"]), src)
            put(f"m7.{key}.unseen", ci(d["eval_unseen_place"]["success"]), src)
            put(f"m7.{key}.seen.obj", ci(d["eval_seen"]["objects"]), src)
            if "eval_unseen_obj" in d:
                put(f"m7.{key}.unseen_obj", ci(d["eval_unseen_obj"]["success"]), src)
            put(f"m7.{key}.min", f"{d['human_min']:.1f}", src)
            put(f"m7.{key}.time", f"{d['time_per_chore_s']:.0f} s" if d["time_per_chore_s"] else "n/a", src)
        for ch in ("sorting", "shelf_putaway", "stacking", "tray_heldout"):
            a, b = A.get(f"{ch}/video_10"), A.get(f"{ch}/teleop_1x")
            if a and b:
                put(f"m7.diff.{ch}", diff(a["eval_seen"]["success"], b["eval_seen"]["success"]), src)
        a, b = A.get("tray_heldout/video_10_multipre"), A.get("tray_heldout/video_10")
        if a and b:
            put("m7.diff.multipre", diff(a["eval_seen"]["success"], b["eval_seen"]["success"]), src)
        M = r["results"].get("part_b", {}).get("matrix", {})
        for k, d in M.items():
            key = k.replace("/", ".")
            put(f"m7.gen.{key}", ci(d["success"]), src)
            put(f"m7.gen.{key}.obj", ci(d["objects"]), src)
    # ---------------- M8
    src = "sim/m8_teach_tray_heldout"
    r = _load(src)
    if r:
        put("m8.minutes", f"{r['results']['human_minutes_total']:.1f}", src)
        tries = r["results"]["state"].get("tries", [])
        if tries:
            put("m8.try_first", ci(tries[0]["success"]), src)
            put("m8.try_last", ci(tries[-1]["success"]), src)
            put("m8.try_last.obj", ci(tries[-1]["objects"]), src)
            put("m8.n_videos", str(tries[-1]["n_videos"]), src)
    # ---------------- safety audit
    src = "sim/safety_audit"
    r = _load(src)
    if r:
        t = r["results"]["total"]
        put("safety.episodes", str(t["episodes"]), src)
        put("safety.ticks", f"{t['ticks']:,}", src)
        put("safety.incidents", str(t["incidents"]), src)
        put("safety.interventions", f"{t['interventions']:,}", src)
        put("safety.max_tcp_speed", f"{t['max_tcp_speed']:.2f}", src)
        put("safety.max_joint_speed_frac", pct(t["max_joint_speed_frac"]), src)
    # ---------------- design ablation (validation)
    src = "sim/dev_policy_design_desk_clearing"
    r = _load(src)
    if r:
        for k, d in r["results"]["rows"].items():
            put(f"dev.{k}", ci(d["objects"]), src)
            put(f"dev.{k}.full", frac(d["success"]), src)
    return {k: v for k, v in R.items() if not k.startswith("_")}


if __name__ == "__main__":
    for k, (v, s) in registry().items():
        print(f"{k:60s} {v:30s} {s}")
