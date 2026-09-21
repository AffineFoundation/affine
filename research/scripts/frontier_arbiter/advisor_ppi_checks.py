"""N4 (ADVISOR gate) and N6 (PPI) $0 checks on stored frontier-arbiter artifacts (2026-09-21).

Check 1 — N4. g(x) = 1 − agreement(informed teacher refs, blind teacher refs) per turn.
  Data: research/results/frontier_arbiter/{privileged_refs,hindsight,hint_design}/ (the 150 D7
  turns; 100 of them carry hint_design cells). Label: the SOURCE trajectory's outcome
  (turns.jsonl down.outcome) — the only outcome available for these turns; none of the 150
  overlaps a continuation-graded state. AUC(g → 'source trajectory FAILED') by variant ×
  agreement metric × dialect, bootstrap CIs, next to controls (blind-vs-blind divergence,
  sampler identity of the source trajectory, note length).

Check 2 — N6. No teacher echo exists for any labelled first action at the 67 continuation-graded
  states (vav/, deliberated/, split_states/, outcome/ carry samples, tables and judge calls, no
  logprobs), so the per-action r = corr(A-leg proxy, V − B) is NOT computable from disk. The
  closest join that exists is reign-level: the live meter's teacher-vs-king control (stored on
  every wvk-22 verdict, shadow.sd_meter.by_anchor.loo.teacher_vs_king) against the verified
  teacher-vs-king pass gap on identical tasks (verified_refs/report.json q2_kings.by_reign) —
  plus the crown chain (live crowning z vs the verified pass1 change between consecutive
  kings). The exact spec + cost of the per-action probe is emitted.

Usage: cd /workspace && source .venv/bin/activate && python research/scripts/frontier_arbiter/advisor_ppi_checks.py
Writes research/results/frontier_arbiter/advisor_ppi/report.{txt,json}. No network, $0.
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import math
import random
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from common import REPO, jaccard, norm_action, read_jsonl

FA = REPO / "research" / "results" / "frontier_arbiter"
OUT = FA / "advisor_ppi"
EVALS = REPO / "research" / "data" / "frontier_arbiter" / "evals"

D7_TURNS = FA / "privileged_refs" / "turns.jsonl"
D7_SAMPLES = FA / "privileged_refs" / "samples.jsonl"
D7_JUDGE = FA / "privileged_refs" / "judge.jsonl"
HS_SAMPLES = FA / "hindsight" / "samples.jsonl"
HS_JUDGE = FA / "hindsight" / "judge.jsonl"
HS_NOTES = FA / "hindsight" / "notes.jsonl"
HD_SAMPLES = FA / "hint_design" / "samples.jsonl"
HD_JUDGE = FA / "hint_design" / "judge.jsonl"
HD_HINTS = FA / "hint_design" / "hints.jsonl"

JAC_THRESH = 0.5
N_BOOT = 2000
SEED = 20260921
DIALECTS = ("bash", "tool_call", "terminus_json")
KILL_AUC = 0.65

# informed variants: (label, sampler, source file family, comparator blind set)
TEACHER_VARIANTS = ("p_hind", "p_err", "h_ground", "h_causal", "h_answer")
KING_VARIANTS = ("h_ground", "h_causal", "h_answer")


# ------------------------------------------------------------------ helpers
def auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank AUC of score for label 1 (ties count 1/2)."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(scores: list[float], labels: list[int], rng: random.Random) -> dict:
    a = auc(scores, labels)
    n = len(scores)
    out = {"n": n, "n_pos": int(sum(labels)), "auc": a, "ci": None}
    if a is None or n < 6:
        return out
    bs = []
    idx = list(range(n))
    for _ in range(N_BOOT):
        pick = [idx[rng.randrange(n)] for _ in range(n)]
        v = auc([scores[i] for i in pick], [labels[i] for i in pick])
        if v is not None:
            bs.append(v)
    bs.sort()
    out["ci"] = [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]]
    out["p_gt_0.5"] = float(np.mean(np.asarray(bs) > 0.5))
    return out


def paired_auc_diff(s1: list[float], s2: list[float], labels: list[int], rng: random.Random) -> dict:
    """Bootstrap of AUC(s1) − AUC(s2) over the same turns."""
    a1, a2 = auc(s1, labels), auc(s2, labels)
    if a1 is None or a2 is None:
        return {"diff": None}
    n = len(labels)
    bs = []
    for _ in range(N_BOOT):
        pick = [rng.randrange(n) for _ in range(n)]
        v1 = auc([s1[i] for i in pick], [labels[i] for i in pick])
        v2 = auc([s2[i] for i in pick], [labels[i] for i in pick])
        if v1 is not None and v2 is not None:
            bs.append(v1 - v2)
    bs.sort()
    return {"diff": a1 - a2, "ci": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if bs else None}


def agreement(informed: list[str], blind: list[str], kind: str) -> dict:
    """Share of informed actions matching the blind set: norm-exact / token-Jaccard >= .5."""
    if not informed or not blind:
        return {"exact": None, "jac50": None, "jac_mean": None}
    nb = [norm_action(b, kind) for b in blind]
    ex = [1.0 if norm_action(y, kind) in nb else 0.0 for y in informed]
    jm = [max(jaccard(y, b) for b in blind) for y in informed]
    return {"exact": st.mean(ex), "jac50": st.mean(1.0 if j >= JAC_THRESH else 0.0 for j in jm), "jac_mean": st.mean(jm)}


def fam(policy_id: str) -> str:
    if policy_id.startswith("king"):
        return "king"
    if policy_id.startswith("glm"):
        return "glm"
    return "teacher"


def _m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def _f(x, w=6, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and x is not None and not (isinstance(x, float) and math.isnan(x)) else " " * (w - 1) + "-"


# ------------------------------------------------------------------ Check 1: g(x) rows
def build_turn_rows() -> tuple[list[dict], dict]:
    turns = read_jsonl(D7_TURNS)
    by_tid = {t["turn_id"]: t for t in turns}
    d7 = collections.defaultdict(list)
    for s in read_jsonl(D7_SAMPLES):
        if s.get("parsed") and s["variant"] == "p_none":
            d7[s["turn_id"]].append(s)
    hs = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in read_jsonl(HS_SAMPLES):
        if s.get("parsed"):
            hs[s["turn_id"]][s["variant"]].append(s)
    hd = collections.defaultdict(lambda: collections.defaultdict(list))
    hd_turns: set[str] = set()
    for s in read_jsonl(HD_SAMPLES):
        hd_turns.add(s["turn_id"])
        if s.get("parsed"):
            hd[s["turn_id"]][(s["sampler"], s["variant"])].append(s)
    notes = {n["turn_id"]: n for n in read_jsonl(HS_NOTES)}
    hints = {h["turn_id"]: h for h in read_jsonl(HD_HINTS)}

    # judge SAME shares: hindsight (informed ref i vs stored blind ref i), hint_design (teacher: vs
    # stored blind; king: vs blind king; blind2 = king 3v3 control), D7 p_none (fresh blind vs stored
    # blind = teacher control) and blind_self (stored ref 1 vs 2).
    judge = collections.defaultdict(list)
    for j in read_jsonl(HS_JUDGE):
        if j.get("same") is not None:
            judge[(j["turn_id"], "teacher", j["variant"])].append(1.0 if j["same"] else 0.0)
    for j in read_jsonl(HD_JUDGE):
        if j.get("same") is not None and j["variant"] != "blind":      # 'blind' pairs = fresh king ref vs y_K (not a divergence)
            judge[(j["turn_id"], j["sampler"], j["variant"])].append(1.0 if j["same"] else 0.0)
    for j in read_jsonl(D7_JUDGE):
        if j.get("same") is not None and j["variant"] in ("p_none", "blind_self"):
            judge[(j["turn_id"], "teacher", j["variant"])].append(1.0 if j["same"] else 0.0)

    rows = []
    for t in turns:
        tid, kind = t["turn_id"], t["dialect"]
        blind_t = [r["y"] for r in t["refs"]]
        fresh_t = [s["y"] for s in sorted(d7.get(tid, []), key=lambda s: s["i"])]
        blind_k = [s["y"] for s in sorted(hd[tid].get(("king", "blind"), []), key=lambda s: s["i"])]
        blind_k2 = [s["y"] for s in sorted(hd[tid].get(("king", "blind2"), []), key=lambda s: s["i"])]
        row = {"turn_id": tid, "dialect": kind, "policy_id": t["policy_id"], "family": fam(t["policy_id"]),
               "source": t["source"], "harness": t["harness"], "outcome": t["down"]["outcome"],
               "failed": 1 if t["down"]["outcome"] == "failed" else 0, "turn_idx": t["turn_idx"],
               "n_prefix_chars": t["n_prefix_chars"], "in_hint_design": tid in hd_turns,
               "g": {}, "g6": {}, "ctrl": {}, "note_words": {}}
        # controls: blind-vs-blind divergence (teacher: fresh p_none vs stored refs; king: blind2 vs blind)
        a = agreement(fresh_t, blind_t, kind)
        row["ctrl"]["teacher_blind"] = {"exact": None if a["exact"] is None else 1 - a["exact"],
                                        "jac50": None if a["jac50"] is None else 1 - a["jac50"],
                                        "judge": None if not judge.get((tid, "teacher", "p_none")) else 1 - st.mean(judge[(tid, "teacher", "p_none")])}
        a = agreement(blind_k2, blind_k, kind)
        row["ctrl"]["king_blind"] = {"exact": None if a["exact"] is None else 1 - a["exact"],
                                     "jac50": None if a["jac50"] is None else 1 - a["jac50"],
                                     "judge": None if not judge.get((tid, "king", "blind2")) else 1 - st.mean(judge[(tid, "king", "blind2")])}
        a = agreement(blind_k, blind_t, kind)
        row["ctrl"]["king_vs_teacher_blind"] = {"exact": None if a["exact"] is None else 1 - a["exact"],
                                                "jac50": None if a["jac50"] is None else 1 - a["jac50"], "judge": None}
        # stored blind self-agreement (within the k=3 stored refs)
        nb = [norm_action(b, kind) for b in blind_t]
        row["ctrl"]["stored_self_exact_div"] = 1 - st.mean(1.0 if nb[i] == nb[j] else 0.0 for i in range(3) for j in range(i + 1, 3)) if len(nb) == 3 else None
        row["ctrl"]["stored_self_judge_div"] = None if not judge.get((tid, "teacher", "blind_self")) else 1 - st.mean(judge[(tid, "teacher", "blind_self")])
        # informed variants
        for v in ("p_hind", "p_err"):
            ys = [s["y"] for s in sorted(hs[tid].get(v, []), key=lambda s: s["i"])]
            if ys:
                a = agreement(ys, blind_t, kind)
                a6 = agreement(ys, blind_t + fresh_t, kind)
                jj = judge.get((tid, "teacher", v))
                row["g"][f"T:{v}"] = {"exact": 1 - a["exact"], "jac50": 1 - a["jac50"], "jac_mean_div": 1 - a["jac_mean"],
                                      "judge": None if not jj else 1 - st.mean(jj), "n_informed": len(ys), "n_judged": len(jj or [])}
                row["g6"][f"T:{v}"] = {"exact": 1 - a6["exact"], "jac50": 1 - a6["jac50"]}
        for v in KING_VARIANTS:
            for sampler, blind in (("teacher", blind_t), ("king", blind_k)):
                ys = [s["y"] for s in sorted(hd[tid].get((sampler, v), []), key=lambda s: s["i"])]
                if ys and blind:
                    a = agreement(ys, blind, kind)
                    jj = judge.get((tid, sampler, v))
                    key = f"{'T' if sampler == 'teacher' else 'K'}:{v}"
                    row["g"][key] = {"exact": 1 - a["exact"], "jac50": 1 - a["jac50"], "jac_mean_div": 1 - a["jac_mean"],
                                     "judge": None if not jj else 1 - st.mean(jj), "n_informed": len(ys), "n_judged": len(jj or [])}
                    if sampler == "teacher":
                        a6 = agreement(ys, blind_t + fresh_t, kind)
                        row["g6"][key] = {"exact": 1 - a6["exact"], "jac50": 1 - a6["jac50"]}
        # note lengths (the hint text itself differs by outcome for p_err / h_causal — a leak channel)
        nrec = notes.get(tid) or {}
        row["note_words"]["p_hind"] = len((nrec.get("p_hind") or "").split()) if nrec.get("p_hind") else None
        row["note_words"]["p_err"] = len((nrec.get("p_err") or "").split()) if nrec.get("p_err") else None
        row["p_err_kind"] = nrec.get("p_err_kind")
        hrec = hints.get(tid) or {}
        for v in KING_VARIANTS:
            hv = (hrec.get("hints") or {}).get(v)
            row["note_words"][v] = len(hv.split()) if hv else None
        rows.append(row)
    meta = {"n_turns": len(rows), "n_hint_design": sum(r["in_hint_design"] for r in rows),
            "by_family_outcome": dict(collections.Counter(f"{r['family']}|{r['outcome']}" for r in rows)),
            "by_dialect_outcome": dict(collections.Counter(f"{r['dialect']}|{r['outcome']}" for r in rows))}
    return rows, meta


def auc_tables(rows: list[dict], rng: random.Random) -> dict:
    subsets = {"all": lambda r: True,
               "teacher_policy": lambda r: r["family"] == "teacher",
               "non_king": lambda r: r["family"] != "king"}
    dialect_sets = {"pooled": None, **{d: d for d in DIALECTS}}
    variants = [f"T:{v}" for v in TEACHER_VARIANTS] + [f"K:{v}" for v in KING_VARIANTS]
    out = {}
    for sname, sfn in subsets.items():
        out[sname] = {}
        for dname, d in dialect_sets.items():
            rs = [r for r in rows if sfn(r) and (d is None or r["dialect"] == d)]
            cell = {"n": len(rs), "n_failed": sum(r["failed"] for r in rs), "variants": {}, "controls": {}, "vs_control": {}}
            for v in variants:
                cell["variants"][v] = {}
                for m in ("exact", "jac50", "judge"):
                    pts = [(r["g"][v][m], r["failed"]) for r in rs if v in r["g"] and r["g"][v].get(m) is not None]
                    cell["variants"][v][m] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
                pts = [(r["g6"][v]["exact"], r["failed"]) for r in rs if v in r["g6"]]
                cell["variants"][v]["exact_vs6blind"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
                pts = [(r["g"][v]["jac_mean_div"], r["failed"]) for r in rs if v in r["g"]]
                cell["variants"][v]["jac_mean"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
                # excess over the matching blind-vs-blind control (same metric, same turns)
                ck = "teacher_blind" if v.startswith("T:") else "king_blind"
                for m in ("exact", "jac50"):
                    pts = [(r["g"][v][m] - r["ctrl"][ck][m], r["failed"]) for r in rs
                           if v in r["g"] and r["g"][v].get(m) is not None and r["ctrl"][ck].get(m) is not None]
                    cell["variants"][v][f"excess_{m}"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
                    trip = [(r["g"][v][m], r["ctrl"][ck][m], r["failed"]) for r in rs
                            if v in r["g"] and r["g"][v].get(m) is not None and r["ctrl"][ck].get(m) is not None]
                    cell["vs_control"][f"{v}|{m}"] = paired_auc_diff([x[0] for x in trip], [x[1] for x in trip], [x[2] for x in trip], rng) if trip else {"diff": None}
            for ck in ("teacher_blind", "king_blind", "king_vs_teacher_blind"):
                cell["controls"][ck] = {}
                for m in ("exact", "jac50", "judge"):
                    pts = [(r["ctrl"][ck][m], r["failed"]) for r in rs if r["ctrl"][ck].get(m) is not None]
                    cell["controls"][ck][m] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
            for ck in ("stored_self_exact_div", "stored_self_judge_div"):
                pts = [(r["ctrl"][ck], r["failed"]) for r in rs if r["ctrl"].get(ck) is not None]
                cell["controls"][ck] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
            pts = [(1.0 if r["family"] == "king" else 0.0, r["failed"]) for r in rs]
            cell["controls"]["is_king_trajectory"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
            pts = [(float(r["turn_idx"]), r["failed"]) for r in rs]
            cell["controls"]["turn_idx"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
            pts = [(float(r["n_prefix_chars"]), r["failed"]) for r in rs]
            cell["controls"]["prefix_chars"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
            for nk in ("p_hind", "p_err", "h_ground", "h_causal", "h_answer"):
                pts = [(float(r["note_words"][nk]), r["failed"]) for r in rs if r["note_words"].get(nk) is not None]
                cell["controls"][f"note_words:{nk}"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
            pts = [(1.0 if r.get("p_err_kind") == "error" else 0.0, r["failed"]) for r in rs if r.get("p_err_kind")]
            cell["controls"]["p_err_has_error"] = boot_auc([p[0] for p in pts], [p[1] for p in pts], rng) if pts else {"n": 0, "auc": None}
            out[sname][dname] = cell
    return out


# ------------------------------------------------------------------ Check 1b: overlap with graded states
def graded_state_overlap(rows: list[dict]) -> dict:
    keys150 = {(r["turn_id"].rsplit(":", 1)[0], r["turn_idx"]) for r in rows}
    turns = read_jsonl(D7_TURNS)
    rk150 = {(t["rollout_id"], t["turn_idx"]) for t in turns}
    conts = read_jsonl(FA / "split_states" / "continuations.jsonl") + read_jsonl(FA / "outcome" / "continuations.jsonl")
    ok = [c for c in conts if c.get("status") == "ok" and c.get("outcome") in ("solved", "failed")]
    states = collections.defaultdict(list)
    for c in ok:
        states[c["state_id"].rsplit(":", 1)[0]].append(c)
    gk = {(s.split(":")[0], int(s.split(":")[1])) for s in states}
    delib = {s["state_id"].rsplit(":", 1)[0] for s in read_jsonl(FA / "deliberated" / "samples.jsonl")}
    dk = {(s.split(":")[0], int(s.split(":")[1])) for s in delib}
    vav = {t["sid"] for t in read_jsonl(FA / "vav" / "tables.jsonl")}
    vk = {(s.split(":")[0], int(s.split(":")[1])) for s in vav}
    return {"n_graded_states": len(states), "n_labelled_first_actions": len(ok),
            "n_graded_states_ge2": sum(1 for v in states.values() if len(v) >= 2),
            "n_graded_states_mixed": sum(1 for v in states.values() if len({c["outcome"] for c in v}) == 2),
            "n_deliberated_states": len(delib), "n_vav_states": len(vav),
            "overlap_150_with_graded": len(rk150 & gk), "overlap_150_with_deliberated": len(rk150 & dk),
            "overlap_150_with_vav": len(rk150 & vk), "informed_samples_at_graded_states": 0,
            "note": "rollout_id:turn_idx keys; the 150 informed-sample turns and the continuation-graded states are disjoint sets"}


# ------------------------------------------------------------------ Check 2: reign-level joins
def load_verdicts() -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(EVALS / "*.json.gz"))):
        d = json.load(gzip.open(f))
        rq, v = d["request"], d["verdict"]
        loo = (((v.get("shadow") or {}).get("sd_meter") or {}).get("by_anchor") or {}).get("loo") or {}
        out.append({"rec": Path(f).name[:10], "king": (rq.get("king_revision") or "")[:12],
                    "chal": (rq.get("challenger_revision") or "")[:12], "wins": v["challenger_wins"],
                    "margin": v["margin"], "se": v["se"], "z": v["z"], "score_mode": v["duel_params"].get("score_mode"),
                    "n_paired": v["n_paired_turns"], "tvk": loo.get("teacher_vs_king"),
                    "teacher_mean": (loo.get("teacher") or {}).get("mean"), "king_mean": (loo.get("king") or {}).get("mean"),
                    "chal_mean": (loo.get("challenger") or {}).get("mean"),
                    "dialects": (v.get("slice") or {}).get("dialects")})
    return out


def per_dialect_control() -> dict:
    """no_typicality/report.json carries per-verdict per-dialect teacher−king means (live variant)."""
    p = FA / "no_typicality" / "report.json"
    if not p.exists():
        return {}
    r = json.load(open(p))
    out = {}
    for v in r.get("verdicts", []):
        pd = ((v.get("variants") or {}).get("live") or {}).get("per_dialect") or {}
        out[v["rec"]] = {k: (x.get("teacher_minus_king") or {}) for k, x in pd.items()}
    return out


def corr(xs: list[float], ys: list[float]) -> dict:
    if len(xs) < 3:
        return {"n": len(xs), "pearson": None, "spearman": None}
    pr = pearsonr(xs, ys)
    sr = spearmanr(xs, ys)
    return {"n": len(xs), "pearson": float(pr[0]), "pearson_p": float(pr[1]), "spearman": float(sr[0]), "spearman_p": float(sr[1])}


def check2() -> dict:
    vr = json.load(open(FA / "verified_refs" / "report.json"))
    by_reign = vr["q2_kings"]["by_reign"]
    dig = {d["digest"][:12]: r for r, d in by_reign.items()}
    verified = {}
    for r, d in by_reign.items():
        vt = d.get("vs_teacher") or {}
        verified[r] = {"digest": d["digest"], "n_tasks": vt.get("n_tasks"), "king_pass1": vt.get("king_pass1"),
                       "teacher_pass1": vt.get("teacher_pass1"), "gap_pass1": vt["teacher_pass1"] - vt["king_pass1"],
                       "gap_passn": vt["teacher_passn"] - vt["king_passn"],
                       "by_group": {g: {"n": x["n"], "gap_passn": x["teacher_passn"] - x["king_passn"]} for g, x in (vt.get("by_group") or {}).items()}}
    verdicts = load_verdicts()
    pdc = per_dialect_control()
    # J2: live teacher-vs-king control (wvk-22 verdicts) vs verified teacher−king pass gap of that king
    j2 = []
    for v in verdicts:
        if not v["tvk"] or v["king"] not in dig:
            continue
        r = dig[v["king"]]
        rec = {"rec": v["rec"], "reign": r, "king": v["king"], "live_tvk_margin": v["tvk"]["margin"], "live_tvk_z": v["tvk"]["z"],
               "live_tvk_se": v["tvk"]["se"], "verified_gap_pass1": verified[r]["gap_pass1"], "verified_gap_passn": verified[r]["gap_passn"],
               "verified_n_tasks": verified[r]["n_tasks"], "per_dialect_live": pdc.get(v["rec"]),
               "verified_by_group": verified[r]["by_group"]}
        j2.append(rec)
    j2_verdict = corr([x["live_tvk_margin"] for x in j2], [x["verified_gap_pass1"] for x in j2])
    j2_verdict_z = corr([x["live_tvk_z"] for x in j2], [x["verified_gap_pass1"] for x in j2])
    by_king = collections.defaultdict(list)
    for x in j2:
        by_king[x["reign"]].append(x)
    j2_king_rows = []
    for r, xs in sorted(by_king.items(), key=lambda kv: int(kv[0].rstrip("r"))):
        j2_king_rows.append({"reign": r, "n_verdicts": len(xs), "live_tvk_margin_mean": st.mean(x["live_tvk_margin"] for x in xs),
                             "live_tvk_z_mean": st.mean(x["live_tvk_z"] for x in xs),
                             "live_sign_teacher_ahead": st.mean(1.0 if x["live_tvk_margin"] > 0 else 0.0 for x in xs),
                             "verified_gap_pass1": xs[0]["verified_gap_pass1"], "verified_gap_passn": xs[0]["verified_gap_passn"],
                             "verified_n_tasks": xs[0]["verified_n_tasks"],
                             "verified_gap_coding": (xs[0]["verified_by_group"].get("coding") or {}).get("gap_passn"),
                             "verified_gap_math": (xs[0]["verified_by_group"].get("math") or {}).get("gap_passn"),
                             "verified_gap_terminal": (xs[0]["verified_by_group"].get("terminal") or {}).get("gap_passn"),
                             "verified_gap_tool_use": (xs[0]["verified_by_group"].get("tool_use") or {}).get("gap_passn"),
                             "live_bash_tmk": _m([((x["per_dialect_live"] or {}).get("bash") or {}).get("mean") for x in xs]),
                             "live_boxed_tmk": _m([((x["per_dialect_live"] or {}).get("boxed") or {}).get("mean") for x in xs]),
                             "live_terminus_tmk": _m([((x["per_dialect_live"] or {}).get("terminus_json") or {}).get("mean") for x in xs]),
                             "live_tool_call_tmk": _m([((x["per_dialect_live"] or {}).get("tool_call") or {}).get("mean") for x in xs])})
    j2_king = corr([x["live_tvk_margin_mean"] for x in j2_king_rows], [x["verified_gap_pass1"] for x in j2_king_rows])
    j2_king_dialect = {}
    for live_k, ver_k in (("live_bash_tmk", "verified_gap_coding"), ("live_boxed_tmk", "verified_gap_math"),
                          ("live_terminus_tmk", "verified_gap_terminal"), ("live_tool_call_tmk", "verified_gap_tool_use")):
        pts = [(x[live_k], x[ver_k]) for x in j2_king_rows if x.get(live_k) is not None and x.get(ver_k) is not None]
        j2_king_dialect[f"{live_k}~{ver_k}"] = corr([p[0] for p in pts], [p[1] for p in pts])
    # pooled dialect↔group points (reign × axis) as one scatter
    pts = []
    for x in j2_king_rows:
        for live_k, ver_k in (("live_bash_tmk", "verified_gap_coding"), ("live_boxed_tmk", "verified_gap_math"),
                              ("live_terminus_tmk", "verified_gap_terminal"), ("live_tool_call_tmk", "verified_gap_tool_use")):
            if x.get(live_k) is not None and x.get(ver_k) is not None:
                pts.append((x[live_k], x[ver_k]))
    j2_king_dialect["pooled_axes"] = corr([p[0] for p in pts], [p[1] for p in pts])
    # J1: crown chain — live crowning z vs verified change (K_new − T) − (K_old − T)
    j1 = []
    for v in verdicts:
        if v["wins"] and v["king"] in dig and v["chal"] in dig:
            ro, rn = dig[v["king"]], dig[v["chal"]]
            d_ver = (-verified[rn]["gap_pass1"]) - (-verified[ro]["gap_pass1"])       # + = new king closer to / above the teacher
            j1.append({"rec": v["rec"], "old_reign": ro, "new_reign": rn, "live_margin": v["margin"], "live_z": v["z"],
                       "score_mode": v["score_mode"], "verified_delta_pass1": d_ver,
                       "old_gap_pass1": verified[ro]["gap_pass1"], "new_gap_pass1": verified[rn]["gap_pass1"],
                       "old_n_tasks": verified[ro]["n_tasks"], "new_n_tasks": verified[rn]["n_tasks"]})
    j1_corr = corr([x["live_z"] for x in j1], [x["verified_delta_pass1"] for x in j1])
    j1_sign = {"n": len(j1), "verified_improved": sum(1 for x in j1 if x["verified_delta_pass1"] > 0),
               "verified_worsened": sum(1 for x in j1 if x["verified_delta_pass1"] < 0)}
    # what can/cannot be joined
    join_notes = [
        "vav/tables.jsonl (18 states, 91 labelled first actions with V) + split_states/outcome continuations (67 states, 240 labelled "
        "first actions) carry outcomes but NO teacher logprobs; vav/, deliberated/, split_states/, outcome/ hold samples, tables, judge "
        "calls and continuation reports only — the per-action live A-leg proxy lpC(y|x,z_own) − lpC(y|x,∅) is not on disk.",
        "verified_refs/report.json is task-level (pass@1 per king reign vs the teacher on identical tasks); it has no per-state or "
        "per-action rows and no live-meter numbers.",
        "no_typicality/report.json + the cached verdicts (chal-00409..00630) carry the live meter per verdict: challenger−king margin "
        "and, on wvk-22 verdicts, the teacher-vs-king control (teacher scored as a miner against the incumbent king on the duel slice) "
        "pooled and per dialect.",
        "The only key both sides share is the KING DIGEST (reigns 11–20). Join J2: live teacher-vs-king control ↔ verified "
        "teacher−king pass gap of that king (7 distinct kings, 35 verdicts = pseudo-replicates). Join J1: crown chain — the live "
        "crowning z of chal→king ↔ the verified pass1 change between consecutive kings (9 crowns). Both are reign-level, both are "
        "'meter vs outcome of a whole POLICY', not 'meter vs verified value of an ACTION at a state' — which is what N6's PPI rectifier "
        "needs. Task mixes also differ per reign (reign 11 = 77% math), so the verified gaps are not on one scale.",
        "Non-crowned challengers have no datagen rollouts (only the king seat is sampled), so the 25 losing wvk-22 challengers cannot be "
        "joined to any verified outcome at all.",
    ]
    # PPI probe spec
    conts = read_jsonl(FA / "split_states" / "continuations.jsonl") + read_jsonl(FA / "outcome" / "continuations.jsonl")
    ok = [c for c in conts if c.get("status") == "ok" and c.get("outcome") in ("solved", "failed")]
    states = collections.defaultdict(list)
    for c in ok:
        states[c["state_id"].rsplit(":", 1)[0]].append(c)
    n_states, n_actions = len(states), len(ok)
    n_actions_with_reply = sum(1 for c in ok if (c.get("first_reply") or {}).get("content"))
    smp = {s["state_id"].rsplit(":", 1)[0]: s for s in read_jsonl(FA / "outcome" / "samples.jsonl")}
    pref = [smp[s]["prefix_chars"] for s in states if s in smp]
    tok_per_state = st.median(pref) / 3.6 if pref else 8000
    echo_price_per_m = 0.03
    n_action_echoes = n_actions_with_reply * 4                  # 3 cross (blind z_i) + 1 empty-context per labelled action
    n_anchor_echoes = n_states * (3 * 3 + 3)                     # refs' cross matrix + empty: LOO μ/σ anchors (live rule)
    n_king_echoes = len(read_jsonl(FA / "vav" / "king_samples.jsonl")) * 4
    usd = (n_action_echoes + n_anchor_echoes + n_king_echoes) * tok_per_state / 1e6 * echo_price_per_m
    spec = {
        "what": "per labelled first action y at graded state x: A(y) = τ·log mean_i exp(b_i/τ) with b_i = [lpC(y|x, z_i) − lpC(y|x, ∅)]·bytes(y), "
                "z_i = the 3 blind teacher thoughts already stored for x in outcome/samples.jsonl (teacher[*].z); live z_A = (A − μ_A)/σ_A with "
                "LOO anchors from the refs' own cross matrix; V − B from the state's continuation table (class = norm_action-exact / Jaccard≥.5).",
        "states": n_states, "labelled_first_actions": n_actions, "with_reply_text": n_actions_with_reply,
        "states_with_ge2_continuations": sum(1 for v in states.values() if len(v) >= 2),
        "states_mixed_outcome": sum(1 for v in states.values() if len({c["outcome"] for c in v}) == 2),
        "blind_refs_available": sum(1 for s in states if s in smp), "prefix_chars_p50": st.median(pref) if pref else None,
        "echoes": {"per_action": 4, "actions": n_action_echoes, "anchors_per_state": 12, "anchors": n_anchor_echoes,
                   "king_samples_vav": n_king_echoes, "total": n_action_echoes + n_anchor_echoes + n_king_echoes},
        "path": "common.TeacherEcho.lp_action (Engy /completions echo, token-id prompt, logprob_start_len) — the prefix is re-materialised "
                "from the stored state json (split_states/outcome `path` fields under /tmp/fa_*; if gone, from the traces manifest by rollout_id)",
        "est_prompt_tokens_per_echo": round(tok_per_state), "price_usd_per_m_prompt": echo_price_per_m,
        "est_cost_usd": round(usd, 2), "est_cost_usd_with_2x_retry_headroom": round(2 * usd, 2),
        "output": "r = corr(z_A(y), V(class(y)) − B(x)) over labelled actions (n≈237), within-state (state-centred) and pooled; per dialect; "
                  "plus AUC(z_A → solved) on the 15 mixed-outcome states; decision rule per §8 N6: r ≤ 0.1 → VAV-with-commit, r < 0 → gate-only.",
        "caveat": "V − B at N = 3–4 per state is itself noisy (§6.5); with 15 mixed states the within-state r has ~50 informative actions — expect a "
                  "wide CI; the harvest at N = 8 (§7) is what makes this rankable.",
    }
    return {"verified_by_reign": verified, "j2_rows": j2, "j2_corr_per_verdict": {"margin": j2_verdict, "z": j2_verdict_z},
            "j2_king_rows": j2_king_rows, "j2_corr_per_king": j2_king, "j2_corr_per_king_dialect_axes": j2_king_dialect,
            "j1_rows": j1, "j1_corr": j1_corr, "j1_sign": j1_sign, "join_notes": join_notes, "ppi_probe_spec": spec,
            "n_verdicts_cached": len(verdicts), "n_verdicts_with_control": len(j2)}


# ------------------------------------------------------------------ report
def fmt_cell(c: dict) -> str:
    if not c or c.get("auc") is None:
        return "     -            "
    ci = c.get("ci")
    return f"{c['auc']:.3f} [{ci[0]:.2f},{ci[1]:.2f}]" if ci else f"{c['auc']:.3f}           "


def write_report(rows, meta, tables, overlap, c2) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    L = []
    P = L.append
    P("N4 (ADVISOR gate) + N6 (PPI) — $0 checks on stored artifacts — 2026-09-21")
    P(f"turns {meta['n_turns']} (hint_design cells on {meta['n_hint_design']}); source-trajectory family|outcome {meta['by_family_outcome']}; dialect|outcome {meta['by_dialect_outcome']}; $ spent 0.00")
    P("")
    P("Terms (one line each):")
    P("  g(x)       = 1 − agreement(informed refs, blind refs) at turn x; informed = teacher (T:) or king (K:) sampled with a hindsight/hint note in the prompt; blind = the verdict's stored k=3 teacher refs (T:) or the fresh k=3 blind king refs (K:).")
    P("  exact      = share of the informed refs whose norm_action (dialect-canonical, whitespace/quotes unified) equals one of the blind refs' — agreement; g = 1 − share.")
    P("  jac50      = share of the informed refs with max token-Jaccard to the blind set ≥ 0.5; g = 1 − share.  jac_mean = 1 − mean of that max Jaccard (continuous).")
    P("  judge      = 1 − share of judged pairs (informed ref i vs blind ref i, i = 0,1; glm-5.3-flash taxonomy) labelled SAME; only on the judged subset (60 hindsight / 40 hint_design turns).")
    P("  exact_vs6  = exact with the blind set widened to the 3 stored + 3 fresh (p_none) blind teacher refs (sensitivity).  excess_* = g − g0 on the same turn.")
    P("  g0 control = the SAME divergence between two blind draws (teacher: fresh p_none vs stored refs; king: blind2 vs blind) = the sampler's own stochasticity at x, no privilege.")
    P("  label      = the SOURCE trajectory (the rollout the prefix was cut from) FAILED its task; AUC = P(g_failed > g_solved) + ½ ties, bootstrap 2000 turn-resamples, 95% percentile CI.")
    P("  p_hind / p_err = hindsight plan note / later-error note (W1); h_ground / h_causal / h_answer = grounding / causal / next-step hints (W3). p_err ('no error… SOLVED' vs an error dump) and h_causal ('the run was solved') state the outcome in the note itself.")
    P("  subsets: all = 150 turns; teacher_policy = the 65 turns whose source trajectory was sampled by the teacher itself (label ≈ 'the teacher failed from here'); non_king = teacher + GLM trajectories (105).")
    P("  live_tvk   = the live meter's teacher-vs-king control on a wvk-22 verdict (teacher scored as a miner against the incumbent king, paired mean over the slice, sd units).  verified gap = teacher_pass1 − king_pass1 on identical datagen tasks for that king (verified_refs Q2).")
    P("")
    P("== CHECK 1 — N4: does blind-vs-informed divergence predict 'source trajectory FAILED'? ==")
    P(f"Overlap with continuation-graded states: 150 informed-sample turns ∩ {overlap['n_graded_states']} graded states = {overlap['overlap_150_with_graded']}; ∩ {overlap['n_deliberated_states']} deliberated states = {overlap['overlap_150_with_deliberated']}; ∩ {overlap['n_vav_states']} VAV states = {overlap['overlap_150_with_vav']}.")
    P("  → NO graded state has informed samples. Every AUC below is against the TRAJECTORY outcome (did the run the prefix came from end solved), not the STATE outcome (does the teacher's blind continuation from x fail).")
    P("    Implication: the label is one bit per trajectory shared by every turn of it, so a turn-level gate cannot be validated here — a turn in a failed run may be fine (the error is later) and a turn in a solved run may be a recovered mistake; and for the 45 king-trajectory turns (44 failed) the label is the KING's failure, not the teacher's.")
    P("")
    variants_all = [f"T:{x}" for x in TEACHER_VARIANTS] + [f"K:{x}" for x in KING_VARIANTS]
    for sname in ("all", "teacher_policy", "non_king"):
        P(f"-- subset {sname}   (AUC [95% CI]; variant × agreement metric × dialect; '-' = no data / one class; CI omitted when n < 6)")
        P(f"{'variant':<12} {'metric':<9} " + " ".join(f"{d:^22}" for d in ("pooled", *DIALECTS)))
        P(f"{'':<12} {'':<9} " + " ".join(f"{'n=' + str(tables[sname][d]['n']) + ' f=' + str(tables[sname][d]['n_failed']):^22}" for d in ("pooled", *DIALECTS)))
        for v in variants_all:
            for m in ("exact", "jac50", "judge", "jac_mean"):
                cells = [tables[sname][d]["variants"][v][m] for d in ("pooled", *DIALECTS)]
                if all(c.get("auc") is None for c in cells):
                    continue
                P(f"{v:<12} {m:<9} " + " ".join(f"{fmt_cell(c):^22}" for c in cells))
        P(f"{'-- controls (no privilege)':<22}")
        for ck in ("teacher_blind", "king_blind", "king_vs_teacher_blind"):
            for m in ("exact", "jac50", "judge"):
                cells = [tables[sname][d]["controls"][ck][m] for d in ("pooled", *DIALECTS)]
                if all(c.get("auc") is None for c in cells):
                    continue
                P(f"{'g0:' + ck[:8]:<12} {m:<9} " + " ".join(f"{fmt_cell(c):^22}" for c in cells))
        for ck in ("stored_self_exact_div", "stored_self_judge_div", "is_king_trajectory", "turn_idx", "prefix_chars", "p_err_has_error",
                   "note_words:p_hind", "note_words:p_err", "note_words:h_ground", "note_words:h_causal", "note_words:h_answer"):
            cells = [tables[sname][d]["controls"][ck] for d in ("pooled", *DIALECTS)]
            if all(c.get("auc") is None for c in cells):
                continue
            P(f"{ck[:21]:<22} " + " ".join(f"{fmt_cell(c):^22}" for c in cells))
        P(f"-- sensitivity (pooled only): exact vs 6-blind union | excess g − g0 (exact / jac50)")
        for v in variants_all:
            c6 = tables[sname]["pooled"]["variants"][v]["exact_vs6blind"]
            ce, cj = tables[sname]["pooled"]["variants"][v]["excess_exact"], tables[sname]["pooled"]["variants"][v]["excess_jac50"]
            if c6.get("auc") is None and ce.get("auc") is None:
                continue
            P(f"{v:<12} exact_vs6 {fmt_cell(c6)}   excess_exact {fmt_cell(ce)}   excess_jac50 {fmt_cell(cj)}")
        P("")
    # headline: max AUC over variants×metrics×dialects, per subset
    P("-- kill test: AUC < 0.65 everywhere?")
    for sname in ("all", "teacher_policy", "non_king"):
        best = []
        for d in ("pooled", *DIALECTS):
            for v, ms in tables[sname][d]["variants"].items():
                for m, c in ms.items():
                    if c.get("auc") is not None and c.get("n", 0) >= 20 and m in ("exact", "jac50", "judge", "jac_mean"):
                        best.append((c["auc"], v, m, d, c["n"], c.get("ci")))
        best.sort(reverse=True)
        top = best[:5]
        ctrl_best = max((tables[sname][d]["controls"][ck][m]["auc"], ck, m, d) for d in ("pooled", *DIALECTS)
                        for ck in ("teacher_blind", "king_blind") for m in ("exact", "jac50", "judge")
                        if tables[sname][d]["controls"][ck][m].get("auc") is not None and tables[sname][d]["controls"][ck][m].get("n", 0) >= 20)
        P(f"  {sname}: top cells (n ≥ 20) " + "; ".join(f"{v} {m} {d} AUC {a:.3f} n {n} CI [{ci[0]:.2f},{ci[1]:.2f}]" for a, v, m, d, n, ci in top))
        P(f"  {sname}: best blind-vs-blind control (no privilege) = {ctrl_best[1]} {ctrl_best[2]} {ctrl_best[3]} AUC {ctrl_best[0]:.3f}; is_king_trajectory AUC {tables[sname]['pooled']['controls']['is_king_trajectory']['auc']:.3f}")
    P("-- informed g vs its own blind control (paired AUC difference, same turns; + = privilege adds signal beyond teacher/king stochasticity)")
    for sname in ("all", "teacher_policy"):
        parts = []
        for v in [f"T:{x}" for x in TEACHER_VARIANTS] + [f"K:{x}" for x in KING_VARIANTS]:
            for m in ("exact", "jac50"):
                d = tables[sname]["pooled"]["vs_control"].get(f"{v}|{m}") or {}
                if d.get("diff") is not None:
                    parts.append(f"{v}/{m} {d['diff']:+.3f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]")
        P(f"  {sname}: " + "; ".join(parts))
    P("")
    P("== CHECK 2 — N6: corr(live meter, verified action value) ==")
    P(f"Per-action r = corr(lpC(y|x,z_own) − lpC(y|x,∅), V − B): NOT COMPUTABLE from disk — 0 teacher echoes exist for any of the {c2['ppi_probe_spec']['states']} graded states / {c2['ppi_probe_spec']['labelled_first_actions']} labelled first actions (checked vav/, deliberated/, split_states/, outcome/).")
    P("What joins and what does not:")
    for n in c2["join_notes"]:
        P(f"  - {n}")
    P("")
    P(f"-- J2: live teacher-vs-king control ↔ verified teacher−king pass1 gap, per king (n kings {len(c2['j2_king_rows'])}, verdicts {c2['n_verdicts_with_control']})")
    P(f"{'reign':<6} {'nV':>3} {'live_tvk_margin':>15} {'live_z':>7} {'P(T ahead)':>10} {'ver_gap_p1':>10} {'ver_n':>6} | {'coding':>7} {'math':>7} {'term':>7} {'tool':>7} | {'bash_tmk':>8} {'boxed_tmk':>9} {'terminus':>8} {'tool_call':>9}")
    for x in c2["j2_king_rows"]:
        P(f"{x['reign']:<6} {x['n_verdicts']:>3} {x['live_tvk_margin_mean']:>15.3f} {x['live_tvk_z_mean']:>7.2f} {x['live_sign_teacher_ahead']:>10.2f} {x['verified_gap_pass1']:>10.3f} {x['verified_n_tasks']:>6} | {_f(x['verified_gap_coding'],7)} {_f(x['verified_gap_math'],7)} {_f(x['verified_gap_terminal'],7)} {_f(x['verified_gap_tool_use'],7)} | {_f(x['live_bash_tmk'],8)} {_f(x['live_boxed_tmk'],9)} {_f(x['live_terminus_tmk'],8)} {_f(x['live_tool_call_tmk'],9)}")
    ck = c2["j2_corr_per_king"]
    cv = c2["j2_corr_per_verdict"]
    P(f"  per king (n={ck['n']}): pearson {_f(ck.get('pearson'))} (p {_f(ck.get('pearson_p'))}) spearman {_f(ck.get('spearman'))}; per verdict (pseudo-replicated, n={cv['margin']['n']}): pearson {_f(cv['margin'].get('pearson'))} spearman {_f(cv['margin'].get('spearman'))} (z: pearson {_f(cv['z'].get('pearson'))})")
    for k, d in c2["j2_corr_per_king_dialect_axes"].items():
        P(f"  dialect↔group axis {k}: n {d['n']} pearson {_f(d.get('pearson'))} spearman {_f(d.get('spearman'))}")
    behind = [x["reign"] for x in c2["j2_king_rows"] if x["live_sign_teacher_ahead"] <= 0.5]
    gaps = [x["verified_gap_pass1"] for x in c2["j2_king_rows"]]
    P(f"  reading: the live control puts the teacher BEHIND the king on reigns {', '.join(behind) or 'none'} (P(T ahead) ≤ 0.5), while every king is verified-below the teacher by {min(gaps):.2f}–{max(gaps):.2f} pass1 on identical tasks — the meter's sign is wrong at the top of the chain even where its ordering across kings is right. tmk = per-dialect teacher−king live mean; coding/math/term/tool = verified teacher−king pass@n by task group.")
    P("")
    P(f"-- J1: crown chain — live crowning z (chal beat king) ↔ verified pass1 change (K_new−T) − (K_old−T)  (n {c2['j1_sign']['n']})")
    P(f"{'rec':<11} {'old':>4} {'new':>4} {'mode':<11} {'live_margin':>11} {'live_z':>7} {'old_gap':>8} {'new_gap':>8} {'Δverified':>10} {'n_old':>6} {'n_new':>6}")
    for x in c2["j1_rows"]:
        P(f"{x['rec']:<11} {x['old_reign']:>4} {x['new_reign']:>4} {str(x['score_mode']):<11} {x['live_margin']:>11.4f} {x['live_z']:>7.2f} {x['old_gap_pass1']:>8.3f} {x['new_gap_pass1']:>8.3f} {x['verified_delta_pass1']:>+10.3f} {x['old_n_tasks']:>6} {x['new_n_tasks']:>6}")
    j1 = c2["j1_corr"]
    P(f"  every crown is live-positive by construction; verified improved {c2['j1_sign']['verified_improved']}/{c2['j1_sign']['n']}, worsened {c2['j1_sign']['verified_worsened']}/{c2['j1_sign']['n']}; corr(live z, Δverified) pearson {_f(j1.get('pearson'))} spearman {_f(j1.get('spearman'))} (n {j1['n']}; units differ across score modes, z is the comparable quantity; task mixes differ per reign).")
    P("")
    s = c2["ppi_probe_spec"]
    P("-- The $-probe that computes N6's r (not run; spec)")
    P(f"  {s['what']}")
    P(f"  inputs on disk: {s['states']} graded states ({s['states_with_ge2_continuations']} with ≥2 continuations, {s['states_mixed_outcome']} with mixed outcomes), {s['labelled_first_actions']} labelled first actions ({s['with_reply_text']} with reply text), blind teacher refs stored for {s['blind_refs_available']}/{s['states']} states, prefix p50 {s['prefix_chars_p50']:.0f} chars ≈ {s['est_prompt_tokens_per_echo']} tokens.")
    P(f"  echoes: {s['echoes']['actions']} action echoes (4 per action: 3 cross + 1 empty) + {s['echoes']['anchors']} anchor echoes (12 per state, LOO μ/σ) + {s['echoes']['king_samples_vav']} for the VAV king samples = {s['echoes']['total']} teacher echoes via {s['path'].split(' (')[0]}.")
    P(f"  cost: ≈ ${s['est_cost_usd']:.2f} at ${s['price_usd_per_m_prompt']}/M prompt tokens (qwen3.8-27b on Engy; echo caching not assumed); ${s['est_cost_usd_with_2x_retry_headroom']:.2f} with 2× retry headroom. Wall ≈ 15–30 min at concurrency 16.")
    P(f"  output: {s['output']}")
    P(f"  caveat: {s['caveat']}")
    P("")
    # ---- data-driven verdict
    tp = tables["teacher_policy"]["pooled"]["variants"]
    al = tables["all"]["pooled"]["variants"]
    leak_free = ("T:p_hind", "T:h_ground", "T:h_answer", "K:h_ground", "K:h_answer")
    outcome_stating = ("T:p_err", "T:h_causal", "K:h_causal")
    def cells_ge(vs, tab, metrics, thr):
        return [(v, m, tab[v][m]["auc"], tab[v][m]["ci"], tab[v][m]["n"]) for v in vs for m in metrics
                if tab[v][m].get("auc") is not None and tab[v][m]["auc"] >= thr]
    lf_tp = cells_ge(leak_free, tp, ("exact", "jac50", "judge", "jac_mean"), KILL_AUC)
    lf_tp_ci = [c for c in lf_tp if c[3] and c[3][0] > 0.5]
    os_tp = cells_ge(outcome_stating, tp, ("exact", "jac50", "judge", "jac_mean"), KILL_AUC)
    exact_max_all = max(al[v]["exact"]["auc"] for v in variants_all if al[v]["exact"].get("auc") is not None)
    exact_max_tp = max(tp[v]["exact"]["auc"] for v in variants_all if tp[v]["exact"].get("auc") is not None)
    ctrl_king = tables["all"]["pooled"]["controls"]["is_king_trajectory"]["auc"]
    n_cells_ge = sum(1 for s in ("all", "teacher_policy", "non_king") for d in ("pooled", *DIALECTS) for v in variants_all
                     for m in ("exact", "jac50", "judge") if tables[s][d]["variants"][v][m].get("auc") is not None
                     and tables[s][d]["variants"][v][m]["n"] >= 20 and tables[s][d]["variants"][v][m]["auc"] >= KILL_AUC)
    n_cells = sum(1 for s in ("all", "teacher_policy", "non_king") for d in ("pooled", *DIALECTS) for v in variants_all
                  for m in ("exact", "jac50", "judge") if tables[s][d]["variants"][v][m].get("auc") is not None
                  and tables[s][d]["variants"][v][m]["n"] >= 20)
    n_cells_ci = sum(1 for s in ("all", "teacher_policy", "non_king") for d in ("pooled", *DIALECTS) for v in variants_all
                     for m in ("exact", "jac50", "judge") if tables[s][d]["variants"][v][m].get("auc") is not None
                     and tables[s][d]["variants"][v][m]["n"] >= 20 and tables[s][d]["variants"][v][m]["auc"] >= KILL_AUC
                     and (tables[s][d]["variants"][v][m].get("ci") or [0])[0] > 0.5)
    n_cells_ci65 = sum(1 for s in ("all", "teacher_policy", "non_king") for d in ("pooled", *DIALECTS) for v in variants_all
                       for m in ("exact", "jac50", "judge") if tables[s][d]["variants"][v][m].get("auc") is not None
                       and tables[s][d]["variants"][v][m]["n"] >= 20 and (tables[s][d]["variants"][v][m].get("ci") or [0])[0] >= KILL_AUC)
    gain_sig = [f"{s}:{k} {d['diff']:+.2f}" for s in ("all", "teacher_policy", "non_king") for k, d in tables[s]["pooled"]["vs_control"].items()
                if d.get("diff") is not None and d.get("ci") and d["ci"][0] > 0]
    P("-- N4 follow-up that would test the actual claim (not run)")
    P("  the claim is about STATE failure; informed samples exist only at trajectory-labelled turns. Reuse the 67 continuation-graded states (V̂(x) = teacher solve rate from x, 15 mixed): write one hindsight note per state from the stored source trajectory (glm-5.3, $0.028/turn as in W1 → ≈ $1.9), sample k = 3 informed teacher refs per state (qwen3.8-27b, ≈ 8k-token prompts → ≈ $0.05), g(x) vs the 3 blind refs already stored (outcome/samples.jsonl); AUC(g → V̂(x) < 0.5) and Spearman(g, V̂). ≈ $2 total; ~67 states is enough to separate AUC 0.5 from 0.75 at 2σ, not finer.")
    P("")
    P("== VERDICT ==")
    P(f"  N4 — not killed by the letter, not supported in substance: {n_cells_ge}/{n_cells} (variant × exact/jac50/judge × dialect × subset, n ≥ 20) cells reach AUC ≥ {KILL_AUC}, {n_cells_ci} of them with a CI excluding 0.5 and {n_cells_ci65} with a CI wholly above {KILL_AUC}; "
      f"on the teacher-policy subset the leak-free variants (p_hind/h_ground/h_answer) clear {KILL_AUC} in {len(lf_tp)} pooled cells ({', '.join(f'{v}/{m} {a:.2f} n{n}' for v, m, a, ci, n in lf_tp) or 'none'}), {len(lf_tp_ci)} with a CI above 0.5; the outcome-stating hints (p_err/h_causal, which say 'solved'/'failed' in the note) clear it in {len(os_tp)} cells — that is the note changing the action, not the teacher knowing where it fails.")
    P(f"  N4 — norm-exact g never exceeds AUC {exact_max_all:.2f} (all) / {exact_max_tp:.2f} (teacher-policy); the paired informed-minus-blind AUC gain (g vs g0, same turns) has a CI excluding 0 in {len(gain_sig)}/{3 * len(tables['all']['pooled']['vs_control'])} subset × variant × metric cells ({', '.join(gain_sig) or 'none'}); on all 150 turns the bit 'source trajectory was a king run' alone scores {ctrl_king:.2f}, above every g cell pooled; and every label is trajectory-level (0/150 turns overlap a graded state), so the gate's actual claim — divergence marks states where the teacher's blind continuation fails — is untested here (≈ $2 spec above).")
    P("  N6 — per-action r = corr(live A-leg proxy, V − B) is NOT computable from disk (0 teacher echoes at the 67 graded states / 240 labelled first actions); the ≈ $0.5–1 echo probe is specified above and is the only way to get the number.")
    cvz = c2["j2_corr_per_king"]
    P(f"  N6 — closest join (reign level, n = {cvz['n']} kings / {c2['n_verdicts_with_control']} verdicts): live teacher-vs-king control ↔ verified teacher−king pass gap, Pearson {_f(cvz.get('pearson'))} / Spearman {_f(cvz.get('spearman'))} per king (n.s.; per-verdict Spearman {_f(c2['j2_corr_per_verdict']['margin'].get('spearman'))} is pseudo-replicated); ordering only weakly positive and the SIGN is wrong at reigns {', '.join(behind)} (meter: king ≥ teacher; verified: teacher ahead by {min(x['verified_gap_pass1'] for x in c2['j2_king_rows'] if x['reign'] in behind):.2f}–{max(x['verified_gap_pass1'] for x in c2['j2_king_rows'] if x['reign'] in behind):.2f} pass1) — RT-7-shaped: as a PPI proxy the live meter is a biased control variate whose rectifier would carry the whole crown at the top of the chain (r ≤ 0.1 → 'VAV with a commit' by §8 N6's own rule); crown chain: verified improved in {c2['j1_sign']['verified_improved']}/{c2['j1_sign']['n']} crowns, corr(live z, Δverified) ρ {_f(c2['j1_corr'].get('spearman'))}.")
    txt = "\n".join(L) + "\n"
    (OUT / "report.txt").write_text(txt)
    js = {"generated_for": "N4 ADVISOR gate + N6 PPI checks", "usd_spent": 0.0, "meta": meta, "overlap": overlap,
          "check1_auc": tables, "check1_rows": rows, "check2": c2}
    (OUT / "report.json").write_text(json.dumps(js, indent=1, default=float))
    print(txt)


def main() -> None:
    rng = random.Random(SEED)
    rows, meta = build_turn_rows()
    tables = auc_tables(rows, rng)
    overlap = graded_state_overlap(rows)
    c2 = check2()
    write_report(rows, meta, tables, overlap, c2)


if __name__ == "__main__":
    main()
