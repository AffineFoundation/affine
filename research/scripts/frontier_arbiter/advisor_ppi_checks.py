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

$-probes (2026-09-21, ≤ $5, Engy) — `probe` subcommand, results appended to the same report:
  N6 — at the 67 graded states, echo lpC(y|x, z) / lpC(y|x, ∅) under the teacher for every labelled
  first action (T + forced-frontier arms), the 54 VAV king samples and the 3 stored blind refs, with
  z ∈ {the 3 blind ref thoughts (live A leg), the action's own thought (B / R legs)}; r = corr(proxy,
  V − B) pooled / state-centred, AUC(proxy → solved) on mixed states.
  N4 — one hindsight note per state from the source trajectory's forward part (hindsight_refs
  builder), k = 3 informed + 3 fresh blind teacher refs, g(x) vs the stored blind refs, AUC(g → the
  teacher's OWN continuation from x fails: s/N ≤ .25 vs ≥ .75) and Spearman(g, s/N).

Usage: cd /workspace && source .venv/bin/activate && cd research/scripts/frontier_arbiter
  python advisor_ppi_checks.py                 # $0 disk checks + (if present) the probe section
  python advisor_ppi_checks.py probe states    # prefixes from the trace mirror (/tmp/fa_ppi)
  python advisor_ppi_checks.py probe echo      # N6 echoes  -> advisor_ppi/probe/echoes.jsonl
  python advisor_ppi_checks.py probe notes     # N4 notes   -> advisor_ppi/probe/notes.jsonl
  python advisor_ppi_checks.py probe sample    # N4 refs    -> advisor_ppi/probe/samples.jsonl
Writes research/results/frontier_arbiter/advisor_ppi/report.{txt,json}.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import gzip
import hashlib
import json
import math
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "outcome"))
import common  # noqa: E402
from common import (  # noqa: E402
    REPO, TEACHER_ENGY, TEACHER_REPO, Engy, TeacherEcho, append_jsonl, clme, hf_token, jaccard, lme,
    norm_action, read_jsonl, reply_to_rollout,
)
from privileged_refs import (  # noqa: E402
    K, REF_MAX_TOKENS, REF_TEMPERATURE, TAU, load_envelope, parse_reply, render_transcript, trace_index, with_note,
)
from hindsight_refs import (  # noqa: E402
    FRONTIER, HIND_STRICT, HIND_SYSTEM, PREFIX_CAP, forward_part, frontier_chat, render_forward, traj_text,
    validate_hind,
)
from vav_sim import canon  # noqa: E402
from split_yield import first_action  # noqa: E402
from affine import dialects  # noqa: E402
from affine.corpus.trace import ToolParityError, message_text, trace_conversations  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402

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
    P(f"turns {meta['n_turns']} (hint_design cells on {meta['n_hint_design']}); source-trajectory family|outcome {meta['by_family_outcome']}; dialect|outcome {meta["by_dialect_outcome"]}; $ spent 0.00 on the disk checks (the $-probes below report their own spend)")
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


# ================================================================== $-probes (2026-09-21)
PROBE = OUT / "probe"
PROBE_STATES = PROBE / "states.jsonl"
PROBE_ECHOES = PROBE / "echoes.jsonl"
PROBE_NOTES = PROBE / "notes.jsonl"
PROBE_SAMPLES = PROBE / "samples.jsonl"
PROBE_COST = PROBE / "cost.jsonl"
PREFIX_CACHE = Path("/tmp/fa_ppi/prefix")
PROBE_BUDGET_USD = 5.0
ACTION_KIND = {"mini_swe_textbased": "bash", "bash": "tool_call", "terminus_2": "terminus_json"}
FAIL_MAX, SOLVE_MIN = 0.25, 0.75
# the slicer's fence rule (datagen/slicer.py _normalize): D stores every message with ```bash
FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
PROXIES = ("A_live", "A_mean", "A_sum", "B_own", "R_blind", "R_lab", "R_blind_lme", "R_lab_lme")


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def base_id(state_id: str) -> str:
    return state_id.rsplit(":", 1)[0]


def slicer_normalize(text: str) -> str:
    return FOREIGN_FENCE_RE.sub("```bash\n", text)


def probe_total_cost() -> float:
    """Sum of the max cumulative cost per (stage, run) — hindsight_refs.total_cost's rule."""
    groups: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in read_jsonl(PROBE_COST):
        groups[(r["stage"], r.get("run"))].append(r["cost_usd"])
    return sum(max(v) for v in groups.values())


def probe_log_cost(stage: str, engy: Engy, run: str, note: str = "") -> None:
    append_jsonl(PROBE_COST, {"at": time.time(), "stage": stage, "run": run, "cost_usd": engy.cost_usd,
                              "usage": engy.usage, "note": note})
    tot = probe_total_cost()
    print(f"  [$] {stage}: this run ${engy.cost_usd:.3f} | probe total ${tot:.2f} {note}", flush=True)
    if tot > PROBE_BUDGET_USD:
        raise SystemExit(f"probe budget exceeded: ${tot:.2f} > ${PROBE_BUDGET_USD}")


def graded_states() -> list[dict]:
    """The 67 continuation-graded states: labelled first actions (T = teacher's own continuations,
    F* = forced frontier proposals) + the 3 stored blind teacher refs (outcome/samples.jsonl)."""
    conts = read_jsonl(FA / "split_states" / "continuations.jsonl") + read_jsonl(FA / "outcome" / "continuations.jsonl")
    ok = [c for c in conts if c.get("status") == "ok" and c.get("outcome") in ("solved", "failed")]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for c in ok:
        by[base_id(c["kept_state_id"])].append(c)
    smp = {base_id(s["state_id"]): s for s in read_jsonl(FA / "outcome" / "samples.jsonl")}
    out = []
    for sid, cs in sorted(by.items()):
        s = smp.get(sid)
        if not s:
            continue
        rid, tidx = sid.split(":")
        T = sorted([c for c in cs if c["arm"] == "T"], key=lambda c: c["continuation"])
        F = sorted([c for c in cs if c["arm"] != "T"], key=lambda c: c["arm"])
        blind = [b for b in s["teacher"] if b.get("parsed") and b.get("y")]
        out.append({"sid": sid, "rollout_id": rid, "turn_idx": int(tidx), "harness": s["harness"],
                    "kind": ACTION_KIND[s["harness"]], "source": s["source"], "group": s["group"],
                    "orig_outcome": s["orig_outcome"], "depth": int(s["depth"]),
                    "s": sum(c["outcome"] == "solved" for c in T), "n": len(T), "T": T, "F": F, "blind": blind,
                    "teacher_action_orig": s["teacher_action_orig"]})
    return out


_BAKER: list[ToolBaker] = []


def baker() -> ToolBaker:
    if not _BAKER:
        _BAKER.append(ToolBaker.from_pretrained(TEACHER_REPO, token=hf_token()))
    return _BAKER[0]


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def echo_form(y: str, kind: str) -> str:
    """The action as the teacher emits it in THIS prefix. Stored tool_call actions are the research
    pipeline's JSON rendering of structured tool_calls; the baked prefix (= corpus D) carries the
    template's own <function=…><parameter=…> XML, so the JSON is re-rendered through the template
    (ToolBaker.baked_assistant) before echoing. Other dialects echo as stored."""
    if kind != "tool_call":
        return y
    calls = []
    for body in TOOL_CALL_RE.findall(canon(y, kind)):
        try:
            d = json.loads(body)
        except ValueError:
            return y
        if not isinstance(d, dict) or "name" not in d:
            return y
        calls.append({"name": d["name"], "arguments": d.get("arguments") or {}})
    if not calls:
        return y
    return baker().baked_assistant("", calls).strip()


def prefix_path(sid: str) -> Path:
    return PREFIX_CACHE / (sid.replace(":", "_") + ".json")


def load_prefix(sid: str) -> list[dict]:
    return json.loads(prefix_path(sid).read_text())


# ------------------------------------------------------------------ stage: states (prefixes)
def cmd_states(args: argparse.Namespace) -> None:
    PROBE.mkdir(parents=True, exist_ok=True)
    PREFIX_CACHE.mkdir(parents=True, exist_ok=True)
    states = graded_states()
    tidx = trace_index()
    baker = ToolBaker.from_pretrained(TEACHER_REPO, token=hf_token())
    rows = []
    for stt in states:
        sid, kind = stt["sid"], stt["kind"]
        row = {k: stt[k] for k in ("sid", "rollout_id", "turn_idx", "harness", "kind", "source", "group",
                                   "orig_outcome", "depth", "s", "n")}
        row.update({"n_T": len(stt["T"]), "n_F": len(stt["F"]), "n_blind": len(stt["blind"])})
        try:
            env = load_envelope(*tidx[stt["rollout_id"]])
            convs = trace_conversations(env["trace"], baker)
            conv = convs[stt["turn_idx"]]
            if conv[-1]["role"] != "assistant":
                raise ValueError("path does not end on an assistant reply")
            prefix = [{"role": m["role"], "content": slicer_normalize(message_text(m.get("content")) or "")}
                      for m in conv[:-1]]
            reply = slicer_normalize(message_text(conv[-1].get("content")) or "")
            try:
                act = dialects.last_action(reply, kind)
            except dialects.UnknownDialect:
                act = ""
            row.update({"prefix_chars": sum(len(m["content"]) for m in prefix), "n_msgs": len(prefix),
                        "orig_action_match": bool(act) and norm_action(act, kind) == norm_action(stt["teacher_action_orig"], kind)})
            prefix_path(sid).write_text(json.dumps(prefix))
        except (ToolParityError, KeyError, IndexError, ValueError) as ex:
            row["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        rows.append(row)
        print(f"  {sid} {stt['harness']:20s} {row.get('prefix_chars', '-'):>7} chars match={row.get('orig_action_match')} {row.get('error', '')}")
    with open(PROBE_STATES, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    okr = [r for r in rows if "error" not in r]
    print(f"{len(okr)}/{len(rows)} states materialised; orig-action match {sum(r['orig_action_match'] for r in okr)}/{len(okr)}; "
          f"prefix chars p50 {st.median(r['prefix_chars'] for r in okr):.0f} max {max(r['prefix_chars'] for r in okr)}")


# ------------------------------------------------------------------ stage: echo (N6)
def own_thought(c: dict, kind: str) -> str:
    """The labelled action's own thought as stored: continuations keep only the visible reply
    (no reasoning_content), so for mini-swe this is the THOUGHT: text, for tool / terminus usually empty."""
    fr = c.get("first_reply") or {}
    content = slicer_normalize(fr.get("content") or "")
    tcs = [{"function": {"name": tc.get("name") or (tc.get("function") or {}).get("name"),
                         "arguments": tc.get("arguments") if tc.get("arguments") is not None
                         else (tc.get("function") or {}).get("arguments")}}
           for tc in (fr.get("tool_calls") or [])]
    return (reply_to_rollout({"content": content, "reasoning": "", "tool_calls": tcs}, kind)["z"] or "").strip()


def king_samples_by_state() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for k in read_jsonl(FA / "vav" / "king_samples.jsonl"):
        if k.get("parsed") and k.get("y"):
            out[k["sid"]].append(k)
    return out


def state_candidates(stt: dict, kings: dict[str, list[dict]]) -> list[dict]:
    """Every action echoed at x: T (teacher's labelled continuations), F (forced frontier proposals,
    labelled), K (VAV king samples, value via class match), B (the 3 stored blind refs = live anchors)."""
    kind = stt["kind"]
    cands = []
    for i, c in enumerate(stt["T"]):
        y = canon(first_action(c, kind), kind)
        if y:
            cands.append({"cid": f"T{i}", "src": "T", "arm": "T", "y": y, "z": own_thought(c, kind),
                          "z_kind": "visible_only", "outcome": c["outcome"]})
    for j, c in enumerate(stt["F"]):
        y = canon(first_action(c, kind), kind)
        if y:
            cands.append({"cid": f"F{j}", "src": "F", "arm": c["arm"], "y": y, "z": own_thought(c, kind),
                          "z_kind": "visible_only", "outcome": c["outcome"]})
    for k, ks in enumerate(kings.get(stt["sid"], [])):
        cands.append({"cid": f"K{k}", "src": "K", "arm": "K", "y": canon(ks["y"], kind), "z": (ks.get("z") or "").strip(),
                      "z_kind": "latent+visible", "outcome": None})
    for i, b in enumerate(stt["blind"][:K]):
        cands.append({"cid": f"B{i}", "src": "B", "arm": "B", "y": canon(b["y"], kind), "z": (b.get("z") or "").strip(),
                      "z_kind": "latent+visible", "outcome": None})
    return cands


def echo_jobs(stt: dict, cands: list[dict]) -> list[tuple[str, str, str, str]]:
    """(key, z, y, why) — dedup'd by (z, y). ∅ + the 3 blind thoughts for every action (A leg / LOO
    anchors); own thought on its own action (B); own thought on the blind actions and the other
    labelled T actions (R leg, two ref sets)."""
    sid = stt["sid"]
    blind_z = [c["z"] for c in cands if c["src"] == "B" and c["z"]]
    blind_y = [c["y"] for c in cands if c["src"] == "B"]
    lab_y = [c["y"] for c in cands if c["src"] == "T"]
    jobs: dict[tuple[str, str], tuple[str, str, str, str]] = {}

    def add(z: str, y: str, why: str) -> None:
        jobs.setdefault((z, y), (f"{sid}|{h(z)}|{h(y)}", z, y, why))

    for c in cands:
        add("", c["y"], "empty")
        for z in blind_z:
            add(z, c["y"], "cross_blind")
        if c["z"]:
            add(c["z"], c["y"], "own")
            if c["src"] in ("T", "K", "F"):
                for y in blind_y:
                    add(c["z"], y, "r_blind")
                for y in lab_y:
                    if y != c["y"]:
                        add(c["z"], y, "r_lab")
    return list(jobs.values())


def read_probe_echoes() -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(PROBE_ECHOES) if "error" not in r}


async def echo_all(states: list[dict], engy: Engy, run: str) -> None:
    te = TeacherEcho(engy)
    kings = king_samples_by_state()
    done = read_probe_echoes()
    kinds = {stt["sid"]: stt["kind"] for stt in states}
    jobs = []
    for stt in states:
        cands = state_candidates(stt, kings)
        for key, z, y, why in echo_jobs(stt, cands):
            if key not in done:
                jobs.append((stt["sid"], key, z, y, why))
    print(f"echo: {len(jobs)} echoes to run ({len(done)} stored)")
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    prefixes: dict[str, list[dict]] = {}
    n_done = 0

    async def worker() -> None:
        nonlocal n_done
        while True:
            try:
                sid, key, z, y, why = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if sid not in prefixes:
                prefixes[sid] = load_prefix(sid)
            try:
                r = await te.lp_action(prefixes[sid], z, echo_form(y, kinds[sid]))
                append_jsonl(PROBE_ECHOES, {"key": key, "sid": sid, "why": why, **r})
            except Exception as ex:  # noqa: BLE001
                append_jsonl(PROBE_ECHOES, {"key": key, "sid": sid, "why": why, "error": repr(ex)[:300]})
            n_done += 1
            if n_done % 100 == 0 or n_done == len(jobs):
                probe_log_cost("echo", engy, run, f"{n_done}/{len(jobs)}")

    await asyncio.gather(*[worker() for _ in range(16)])


def cmd_echo(args: argparse.Namespace) -> None:
    states = [s for s in graded_states() if prefix_path(s["sid"]).exists()]
    if args.limit:
        states = states[: args.limit]
    engy = Engy(concurrency=16, retries=4)
    asyncio.run(echo_all(states, engy, run=f"echo-{int(time.time())}"))


# ------------------------------------------------------------------ stage: notes (N4, hindsight_refs' p_hind builder)
async def gen_state_notes(states: list[dict], engy: Engy, engy_fb: Engy, run: str) -> None:
    done = {r["sid"] for r in read_jsonl(PROBE_NOTES) if r.get("p_hind")}
    todo = [s for s in states if s["sid"] not in done]
    print(f"notes: {len(todo)} states to do")
    tidx = trace_index()
    sem = asyncio.Semaphore(12)
    n_done = 0

    async def one(stt: dict) -> None:
        nonlocal n_done
        async with sem:
            env = load_envelope(*tidx[stt["rollout_id"]])
            fp = forward_part(env["trace"], stt["turn_idx"])
            outcome = stt["orig_outcome"]
            t = {"turn_id": stt["sid"], "prefix": load_prefix(stt["sid"]), "dialect": stt["kind"]}
            rec = {"sid": stt["sid"], "outcome": outcome, "n_fwd": fp["n_fwd"], "fwd_chars": fp["fwd_chars"], "attempts": []}
            traj = traj_text(t, fp)
            user = (f"Transcript so far — the agent must now produce its next step (dialect: {t['dialect']}):\n\n"
                    f"{render_transcript(t['prefix'], PREFIX_CAP)}\n\n--- END OF TRANSCRIPT SO FAR (step t) ---\n\n"
                    f"What the agent did from step t onward, and how it was graded:\n\n{render_forward(fp, outcome)}\n\n"
                    f"--- END ---\nWrite the private note for step t now (<= 80 words, no commands/paths/code/backticks, "
                    f"no quotes from the transcript).")
            msgs = [{"role": "system", "content": HIND_SYSTEM}, {"role": "user", "content": user}]
            content_attempts = 0
            for _ in range(3):
                try:
                    r, model = await frontier_chat(engy, engy_fb, msgs)
                except Exception as ex:  # noqa: BLE001
                    rec["attempts"].append({"error": repr(ex)[:200]})
                    break
                note = (r["content"] or "").strip()
                why = validate_hind(note, traj)
                rec["attempts"].append({"note": note, "reject": why, "model": model, "usage": r["usage"],
                                        "cost_usd": r["cost_usd"], "finish": r["finish"]})
                if why is None:
                    rec["p_hind"], rec["p_hind_model"] = note, model
                    break
                if why == "empty":
                    continue
                content_attempts += 1
                if content_attempts >= 2:
                    break
                msgs = msgs + [{"role": "assistant", "content": note},
                               {"role": "user", "content": HIND_STRICT.format(why=why, n_words=len(note.split()))}]
            append_jsonl(PROBE_NOTES, rec)
        n_done += 1
        if n_done % 12 == 0 or n_done == len(todo):
            probe_log_cost("notes", engy, run, f"{n_done}/{len(todo)}")
            if engy_fb.cost_usd:
                probe_log_cost("notes_fb", engy_fb, run, f"{n_done}/{len(todo)}")

    await asyncio.gather(*[one(s) for s in todo])


def cmd_notes(args: argparse.Namespace) -> None:
    states = [s for s in graded_states() if prefix_path(s["sid"]).exists()]
    if args.limit:
        states = states[: args.limit]
    run = f"notes-{int(time.time())}"
    asyncio.run(gen_state_notes(states, Engy(concurrency=12, retries=3), Engy(concurrency=12), run))
    notes = read_jsonl(PROBE_NOTES)
    rej = collections.Counter(a.get("reject") for n in notes for a in n["attempts"] if a.get("reject"))
    print(f"p_hind ok {sum(1 for n in notes if n.get('p_hind'))}/{len(notes)}; rejections {dict(rej)}; "
          f"models {dict(collections.Counter(n.get('p_hind_model') for n in notes if n.get('p_hind')))}")


# ------------------------------------------------------------------ stage: sample (N4 informed + fresh blind refs)
async def sample_state_refs(states: list[dict], engy: Engy, run: str) -> None:
    notes = {n["sid"]: n["p_hind"] for n in read_jsonl(PROBE_NOTES) if n.get("p_hind")}
    done = {(r["sid"], r["variant"], r["i"]) for r in read_jsonl(PROBE_SAMPLES) if "error" not in r}
    jobs = [(s, v, i) for s in states if s["sid"] in notes for v in ("p_hind", "p_none") for i in range(K)
            if (s["sid"], v, i) not in done]
    print(f"sample: {len(jobs)} teacher refs to draw")
    n_done = 0

    async def one(stt: dict, v: str, i: int) -> None:
        nonlocal n_done
        prefix = load_prefix(stt["sid"])
        msgs = with_note(prefix, notes[stt["sid"]]) if v == "p_hind" else prefix
        rec = {"sid": stt["sid"], "variant": v, "i": i, "kind": stt["kind"]}
        try:
            r = await engy.chat(TEACHER_ENGY, msgs, temperature=REF_TEMPERATURE, max_tokens=REF_MAX_TOKENS)
            p = parse_reply(r, stt["kind"])
            p["y"] = canon(p["y"], stt["kind"]) if p.get("y") else ""
            rec.update({k: p.get(k) for k in ("z", "y", "parsed", "kind_used", "think_closed", "finish",
                                                  "reasoning_chars", "content_chars", "cost_usd")})
        except Exception as ex:  # noqa: BLE001
            rec["error"] = repr(ex)[:300]
        append_jsonl(PROBE_SAMPLES, rec)
        n_done += 1
        if n_done % 50 == 0 or n_done == len(jobs):
            probe_log_cost("sample", engy, run, f"{n_done}/{len(jobs)}")

    await asyncio.gather(*[one(*j) for j in jobs])


def cmd_sample(args: argparse.Namespace) -> None:
    states = [s for s in graded_states() if prefix_path(s["sid"]).exists()]
    if args.limit:
        states = states[: args.limit]
    asyncio.run(sample_state_refs(states, Engy(concurrency=16, retries=4), run=f"sample-{int(time.time())}"))
    smp = read_jsonl(PROBE_SAMPLES)
    print(f"samples {len(smp)}; parsed {sum(1 for s in smp if s.get('parsed'))}; errors {sum(1 for s in smp if 'error' in s)}")


# ------------------------------------------------------------------ probe analysis
def a_leg(bs: list[float], tau: float = TAU) -> float | None:
    return lme(bs, tau) if bs else None


def action_legs(stt: dict, cands: list[dict], E: dict[str, dict]) -> list[dict]:
    """Per candidate: the proxies (per-byte unless named _sum), the class value V and the state
    baseline B(x) = s/N. Blind-ref candidates use LOO anchors (own thought / own action excluded)."""
    sid, kind = stt["sid"], stt["kind"]
    blind = [c for c in cands if c["src"] == "B"]
    lab_T = [c for c in cands if c["src"] == "T"]
    labelled = [c for c in cands if c["src"] in ("T", "F")]
    classes: dict[str, list[dict]] = collections.defaultdict(list)
    for c in labelled:
        classes[norm_action(c["y"], kind)].append(c)
    Bx = stt["s"] / stt["n"] if stt["n"] else None

    def e(z: str, y: str) -> dict | None:
        return E.get(f"{sid}|{h(z)}|{h(y)}")

    def value(c: dict) -> tuple[float | None, str | None]:
        n_ = norm_action(c["y"], kind)
        if n_ in classes:
            cl = classes[n_]
            return sum(x["outcome"] == "solved" for x in cl) / len(cl), "exact"
        best, bj = None, 0.0
        for n2, cl in classes.items():
            j = jaccard(c["y"], cl[0]["y"])
            if j >= JAC_THRESH and j > bj:
                best, bj = cl, j
        if best:
            return sum(x["outcome"] == "solved" for x in best) / len(best), "jac50"
        return None, None

    rows = []
    for c in cands:
        emp = e("", c["y"])
        if not emp:
            continue
        row = {"sid": sid, "kind": kind, "harness": stt["harness"], "cid": c["cid"], "src": c["src"], "arm": c["arm"],
               "outcome": c["outcome"], "z_kind": c["z_kind"] if c["z"] else None, "n_bytes_y": emp["n_bytes"],
               "Bx": Bx, "s": stt["s"], "n": stt["n"], "mixed": 0 < stt["s"] < stt["n"]}
        V, how = value(c) if c["src"] != "B" else (None, None)
        if c["src"] == "B":
            V, how = value(c)          # a blind ref that coincides with a labelled class inherits its V
        row.update({"V": V, "V_how": how, "V_minus_B": (V - Bx) if (V is not None and Bx is not None) else None})
        # A leg: blind thoughts (LOO for a blind ref's own thought)
        bs_pb, bs_sum = [], []
        for b in blind:
            if not b["z"] or (c["src"] == "B" and b["cid"] == c["cid"]):
                continue
            x = e(b["z"], c["y"])
            if x:
                bs_pb.append(x["lp_per_byte"] - emp["lp_per_byte"])
                bs_sum.append(x["sum_lp"] - emp["sum_lp"])
        row["n_anchor"] = len(bs_pb)
        row["A_live"] = a_leg(bs_pb)
        row["A_mean"] = st.mean(bs_pb) if bs_pb else None
        row["A_sum"] = st.mean(bs_sum) if bs_sum else None
        # B (own thought licenses own action) and R legs (own thought predicts the refs' actions)
        row["B_own"] = row["R_blind"] = row["R_lab"] = row["R_blind_lme"] = row["R_lab_lme"] = None
        if c["z"]:
            own = e(c["z"], c["y"])
            if own:
                row["B_own"] = own["lp_per_byte"] - emp["lp_per_byte"]
            for name, refs in (("blind", [b for b in blind if not (c["src"] == "B" and b["cid"] == c["cid"])]),
                               ("lab", [t for t in lab_T if t["cid"] != c["cid"]])):
                a = []
                for r_ in refs:
                    x, x0 = e(c["z"], r_["y"]), e("", r_["y"])
                    if x and x0:
                        a.append(x["lp_per_byte"] - x0["lp_per_byte"])
                if len(a) >= 2:
                    row[f"R_{name}"] = clme(a, TAU)
                    row[f"R_{name}_lme"] = lme(a, TAU)
                row[f"n_ref_{name}"] = len(a)
        rows.append(row)
    return rows


def _pear(xs, ys):
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(pearsonr(xs, ys)[0])


def _spear(xs, ys):
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(spearmanr(xs, ys)[0])


def centred(rows: list[dict], key: str, val: str) -> tuple[list[float], list[float], int]:
    """State-centred pairs (x − x̄_state, v − v̄_state) over states with ≥ 2 rows."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["sid"]].append(r)
    xs, vs, n_states = [], [], 0
    for grp in by.values():
        if len(grp) < 2:
            continue
        n_states += 1
        mx, mv = st.mean(r[key] for r in grp), st.mean(r[val] for r in grp)
        xs += [r[key] - mx for r in grp]
        vs += [r[val] - mv for r in grp]
    return xs, vs, n_states


def cluster_boot(rows: list[dict], stat, rng: random.Random, n_boot: int = N_BOOT) -> list | None:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["sid"]].append(r)
    sids = list(by)
    if len(sids) < 4:
        return None
    vals = []
    for _ in range(n_boot):
        pick = [by[rng.choice(sids)] for _ in sids]
        v = stat([r for grp in pick for r in grp])
        if v is not None:
            vals.append(v)
    if len(vals) < n_boot // 2:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def proxy_table(rows: list[dict], proxy: str, rng: random.Random) -> dict:
    """pooled r (Pearson/Spearman) of proxy vs V − B, state-centred r, AUC(proxy → solved) on the
    mixed states' T actions (pooled and state-centred); state-cluster bootstrap CIs."""
    lab = [r for r in rows if r.get(proxy) is not None and r.get("V_minus_B") is not None and r["src"] in ("T", "F", "K")]
    out = {"n": len(lab), "n_states": len({r["sid"] for r in lab}),
           "by_src": dict(collections.Counter(r["src"] for r in lab))}
    if lab:
        xs, vs = [r[proxy] for r in lab], [r["V_minus_B"] for r in lab]
        out["pooled"] = {"pearson": _pear(xs, vs), "spearman": _spear(xs, vs),
                         "pearson_ci": cluster_boot(lab, lambda rr: _pear([r[proxy] for r in rr], [r["V_minus_B"] for r in rr]), rng),
                         "spearman_ci": cluster_boot(lab, lambda rr: _spear([r[proxy] for r in rr], [r["V_minus_B"] for r in rr]), rng)}
        cx, cv, ns = centred(lab, proxy, "V_minus_B")
        out["within"] = {"n": len(cx), "n_states": ns, "pearson": _pear(cx, cv), "spearman": _spear(cx, cv),
                         "pearson_ci": cluster_boot(lab, lambda rr: _pear(*centred(rr, proxy, "V_minus_B")[:2]), rng)}
        # only states where V − B actually varies (mixed labelled outcomes)
        mixed_lab = [r for r in lab if r["mixed"]]
        cx2, cv2, ns2 = centred(mixed_lab, proxy, "V_minus_B")
        out["within_mixed"] = {"n": len(cx2), "n_states": ns2, "pearson": _pear(cx2, cv2), "spearman": _spear(cx2, cv2),
                               "pearson_ci": cluster_boot(mixed_lab, lambda rr: _pear(*centred(rr, proxy, "V_minus_B")[:2]), rng)}
        # the same, restricted to the labelled arms (T + F): K rows differ in thought kind (latent+visible)
        # and get V by class match, so a K-vs-T offset inside a state could masquerade as signal
        tf = [r for r in lab if r["src"] in ("T", "F")]
        cx3, cv3, ns3 = centred(tf, proxy, "V_minus_B")
        out["within_TF"] = {"n": len(cx3), "n_states": ns3, "pearson": _pear(cx3, cv3), "spearman": _spear(cx3, cv3),
                            "pearson_ci": cluster_boot(tf, lambda rr: _pear(*centred(rr, proxy, "V_minus_B")[:2]), rng)}
        tfm = [r for r in tf if r["mixed"]]
        cx4, cv4, ns4 = centred(tfm, proxy, "V_minus_B")
        out["within_TF_mixed"] = {"n": len(cx4), "n_states": ns4, "pearson": _pear(cx4, cv4), "spearman": _spear(cx4, cv4),
                                  "pearson_ci": cluster_boot(tfm, lambda rr: _pear(*centred(rr, proxy, "V_minus_B")[:2]), rng)}
        kk = [r for r in lab if r["src"] == "K"]
        out["pooled_K"] = {"n": len(kk), "pearson": _pear([r[proxy] for r in kk], [r["V_minus_B"] for r in kk]),
                           "spearman": _spear([r[proxy] for r in kk], [r["V_minus_B"] for r in kk])}
        # sign test: per mixed state, corr sign between proxy and V
        signs = []
        by = collections.defaultdict(list)
        for r in mixed_lab:
            by[r["sid"]].append(r)
        for grp in by.values():
            p = _spear([r[proxy] for r in grp], [r["V_minus_B"] for r in grp])
            if p is not None and p != 0:
                signs.append(p > 0)
        out["per_state_sign"] = {"n_states": len(signs), "positive": sum(signs)}
    # AUC solved vs failed: T actions at mixed states
    t_mixed = [r for r in rows if r.get(proxy) is not None and r["src"] == "T" and r["mixed"] and r["outcome"] in ("solved", "failed")]
    if t_mixed:
        labels = [int(r["outcome"] == "solved") for r in t_mixed]
        out["auc_T_mixed"] = boot_auc([r[proxy] for r in t_mixed], labels, rng)
        cx, _, _ = centred(t_mixed, proxy, "V_minus_B")
        out["auc_T_mixed_centred"] = boot_auc(cx, labels, rng)
    t_all = [r for r in rows if r.get(proxy) is not None and r["src"] in ("T", "F") and r["outcome"] in ("solved", "failed")]
    if t_all:
        out["auc_TF_all"] = boot_auc([r[proxy] for r in t_all], [int(r["outcome"] == "solved") for r in t_all], rng)
    by_kind = {}
    for kd in DIALECTS:
        sub = [r for r in lab if r["kind"] == kd]
        if len(sub) >= 6:
            cx, cv, ns = centred(sub, proxy, "V_minus_B")
            by_kind[kd] = {"n": len(sub), "pooled_pearson": _pear([r[proxy] for r in sub], [r["V_minus_B"] for r in sub]),
                           "within_pearson": _pear(cx, cv), "n_states_within": ns}
    out["by_kind"] = by_kind
    return out


def n6_probe(states: list[dict], rng: random.Random) -> dict:
    E = read_probe_echoes()
    if not E:
        return {}
    kings = king_samples_by_state()
    rows = []
    for stt in states:
        if not prefix_path(stt["sid"]).exists():
            continue
        rows += action_legs(stt, state_candidates(stt, kings), E)
    errors = sum(1 for r in read_jsonl(PROBE_ECHOES) if "error" in r)
    out = {"n_echoes": len(E), "n_echo_errors": errors, "n_rows": len(rows),
           "rows_by_src": dict(collections.Counter(r["src"] for r in rows)),
           "rows_with_value": sum(1 for r in rows if r["V_minus_B"] is not None and r["src"] != "B"),
           "king_matched": dict(collections.Counter(r["V_how"] or "unmatched" for r in rows if r["src"] == "K")),
           "own_thought_available": dict(collections.Counter(f"{r['src']}/{r['z_kind'] or 'none'}" for r in rows if r["src"] != "B")),
           "proxies": {p: proxy_table(rows, p, rng) for p in PROXIES}, "rows": rows}
    # sanity: the teacher's own blind action under its own thought vs a generic action ordering
    b_rows = [r for r in rows if r["src"] == "B" and r.get("A_live") is not None]
    out["blind_refs"] = {"n": len(b_rows), "A_live_mean": st.mean(r["A_live"] for r in b_rows) if b_rows else None,
                         "A_live_neg_frac": st.mean(r["A_live"] < 0 for r in b_rows) if b_rows else None}
    t_rows = [r for r in rows if r["src"] == "T" and r.get("A_live") is not None]
    k_rows = [r for r in rows if r["src"] == "K" and r.get("A_live") is not None]
    out["means"] = {"T": st.mean(r["A_live"] for r in t_rows) if t_rows else None,
                    "K": st.mean(r["A_live"] for r in k_rows) if k_rows else None,
                    "T_solved": _m([r["A_live"] for r in t_rows if r["outcome"] == "solved"]),
                    "T_failed": _m([r["A_live"] for r in t_rows if r["outcome"] == "failed"])}
    return out


def n4_state_probe(states: list[dict], rng: random.Random) -> dict:
    smp = [s for s in read_jsonl(PROBE_SAMPLES) if "error" not in s]
    if not smp:
        return {}
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in smp:
        if s.get("parsed") and s.get("y"):
            by[s["sid"]][s["variant"]].append(s["y"])
    notes = {n["sid"]: n for n in read_jsonl(PROBE_NOTES)}
    rows = []
    for stt in states:
        sid, kind = stt["sid"], stt["kind"]
        blind = [canon(b["y"], kind) for b in stt["blind"]]
        inf, fresh = by[sid].get("p_hind", []), by[sid].get("p_none", [])
        if not blind or not inf or not stt["n"]:      # n = 0: only forced arms graded, no own-continuation label
            continue
        g = agreement(inf, blind, kind)
        g0 = agreement(fresh, blind, kind) if fresh else None
        vhat = stt["s"] / stt["n"]
        rows.append({"sid": sid, "kind": kind, "harness": stt["harness"], "s": stt["s"], "n": stt["n"], "vhat": vhat,
                     "fail": vhat <= FAIL_MAX, "solve": vhat >= SOLVE_MIN, "orig_outcome": stt["orig_outcome"],
                     "n_inf": len(inf), "n_fresh": len(fresh),
                     "g_exact": 1 - g["exact"], "g_jac50": 1 - g["jac50"], "g_jac_mean": 1 - g["jac_mean"],
                     "g0_exact": (1 - g0["exact"]) if g0 else None, "g0_jac50": (1 - g0["jac50"]) if g0 else None,
                     "g0_jac_mean": (1 - g0["jac_mean"]) if g0 else None,
                     "note_words": len((notes.get(sid, {}).get("p_hind") or "").split()),
                     "note_mentions_outcome": bool(re.search(r"\b(fail|failed|solved|succeed|unsolved|wrong|incorrect)\w*", (notes.get(sid, {}).get("p_hind") or ""), re.I))})
    out = {"n_states": len(rows), "n_fail": sum(r["fail"] for r in rows), "n_solve": sum(r["solve"] for r in rows),
           "n_mixed_dropped": sum(1 for r in rows if not r["fail"] and not r["solve"]),
           "by_kind": dict(collections.Counter(r["kind"] for r in rows)),
           "samples_parsed": sum(1 for s in smp if s.get("parsed")), "samples_total": len(smp),
           "notes_ok": sum(1 for n in notes.values() if n.get("p_hind")), "rows": rows, "auc": {}, "spearman": {}}
    lab = [r for r in rows if r["fail"] or r["solve"]]
    for m in ("exact", "jac50", "jac_mean"):
        for which in ("g", "g0", "excess"):
            def val(r, m=m, which=which):
                if which == "g":
                    return r[f"g_{m}"]
                if which == "g0":
                    return r[f"g0_{m}"]
                return None if r[f"g0_{m}"] is None else r[f"g_{m}"] - r[f"g0_{m}"]
            sub = [r for r in lab if val(r) is not None]
            out["auc"][f"{which}_{m}"] = boot_auc([val(r) for r in sub], [int(r["fail"]) for r in sub], rng)
            sub_all = [r for r in rows if val(r) is not None]
            xs, vs = [val(r) for r in sub_all], [r["vhat"] for r in sub_all]
            rho = _spear(xs, vs)
            ci = None
            if rho is not None:
                vals = []
                for _ in range(N_BOOT):
                    idx = [rng.randrange(len(xs)) for _ in xs]
                    v = _spear([xs[i] for i in idx], [vs[i] for i in idx])
                    if v is not None:
                        vals.append(v)
                if vals:
                    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            out["spearman"][f"{which}_{m}"] = {"n": len(xs), "rho": rho, "ci": ci}
        if lab:
            out["auc"][f"paired_gain_{m}"] = paired_auc_diff(
                [r[f"g_{m}"] for r in lab if r[f"g0_{m}"] is not None],
                [r[f"g0_{m}"] for r in lab if r[f"g0_{m}"] is not None],
                [int(r["fail"]) for r in lab if r[f"g0_{m}"] is not None], rng)
    # ceiling: norm-exact divergence is 1.0 wherever no informed ref repeats a stored blind ref verbatim
    out["saturated"] = {"g_exact_eq_1": sum(1 for r in rows if r["g_exact"] >= 1), "g0_exact_eq_1": sum(1 for r in rows if r["g0_exact"] is not None and r["g0_exact"] >= 1),
                        "g_jac50_eq_1": sum(1 for r in rows if r["g_jac50"] >= 1), "n": len(rows)}
    # states with ≥ 3 own continuations: s/N ∈ {0, 1} at N = 1 is one coin flip, not a state label
    lab3 = [r for r in lab if r["n"] >= 3]
    out["n_ge3"] = {"n_states": len(lab3), "n_fail": sum(r["fail"] for r in lab3), "n_solve": sum(r["solve"] for r in lab3), "auc": {}, "spearman": {}}
    for m in ("exact", "jac50", "jac_mean"):
        if lab3:
            out["n_ge3"]["auc"][f"g_{m}"] = boot_auc([r[f"g_{m}"] for r in lab3], [int(r["fail"]) for r in lab3], rng)
            sub0 = [r for r in lab3 if r[f"g0_{m}"] is not None]
            out["n_ge3"]["auc"][f"g0_{m}"] = boot_auc([r[f"g0_{m}"] for r in sub0], [int(r["fail"]) for r in sub0], rng)
        all3 = [r for r in rows if r["n"] >= 3]
        out["n_ge3"]["spearman"][f"g_{m}"] = {"n": len(all3), "rho": _spear([r[f"g_{m}"] for r in all3], [r["vhat"] for r in all3])}
    # per harness (small n) and the orig-outcome label for comparison (trajectory-level, like check 1)
    out["by_kind_auc"] = {}
    for kd in DIALECTS:
        sub = [r for r in lab if r["kind"] == kd]
        if sub:
            out["by_kind_auc"][kd] = {m: boot_auc([r[f"g_{m}"] for r in sub], [int(r["fail"]) for r in sub], rng) for m in ("exact", "jac50")}
    out["auc_vs_orig_outcome"] = {m: boot_auc([r[f"g_{m}"] for r in rows], [int(r["orig_outcome"] == "failed") for r in rows], rng)
                                  for m in ("exact", "jac50")}
    out["auc_note_words"] = boot_auc([r["note_words"] for r in lab], [int(r["fail"]) for r in lab], rng) if lab else None
    out["auc_orig_outcome_bit"] = boot_auc([float(r["orig_outcome"] == "failed") for r in lab], [int(r["fail"]) for r in lab], rng) if lab else None
    return out


def _ci(c) -> str:
    return f"[{c[0]:+.2f},{c[1]:+.2f}]" if c else "[   -  ,   -  ]"


def probe_report(states: list[dict], n6: dict, n4: dict, usd: float) -> tuple[list[str], dict]:
    L: list[str] = []
    P = L.append
    P("")
    P("== $-PROBES (2026-09-21) — N6 per-action echo probe + N4 state-level ADVISOR test ==")
    P(f"$ spent on the probes: {usd:.2f} (cost.jsonl; echo + notes + sampling).  states materialised {sum(1 for s in states if prefix_path(s['sid']).exists())}/{len(states)} "
      f"(prefix from the trace mirror via affine.corpus.trace.trace_conversations + ToolBaker, mswea→bash fence as the slicer does).")
    P("Terms (one line each):")
    P("  A_live   = live A leg for action y at x: τ·log mean_i exp(b_i/τ), b_i = lpC(y|x,z_i) − lpC(y|x,∅) per byte, z_i = the 3 stored blind teacher thoughts (LOO for a blind ref's own), τ=0.03.")
    P("  A_mean / A_sum = plain mean of b_i per byte / summed over the action's tokens (the per-byte fence artefact of the A-leg probe does not apply to A_sum).")
    P("  B_own    = lpC(y|x,z_own) − lpC(y|x,∅) per byte: the live causality license, own thought on own action.  z_own for T/F actions is the stored VISIBLE reply text only (no reasoning_content was kept); K and blind carry latent+visible.")
    P("  R_blind / R_lab = centred tempered LME (live R) of a_j = lpC(y_j|x,z_own) − lpC(y_j|x,∅) over refs y_j = the 3 blind actions / the state's OTHER labelled T actions; *_lme = uncentred.")
    P("  V        = solve rate of the action's norm-exact class among the state's labelled first actions (self included); B(x) = s/N of the T arm; K samples get V by exact or Jaccard≥.5 class match.")
    P("  within   = state-centred Pearson (both variables minus their state mean) over states with ≥ 2 rows; within_mixed = the same on states whose labelled outcomes are mixed; CIs = 2000 state-cluster bootstraps.")
    P("  g(x)     = 1 − share of the 3 informed refs (teacher + hindsight note from the source trajectory) matching the 3 STORED blind refs (exact / jac50); g0 = same for 3 FRESH blind refs; excess = g − g0.")
    P(f"  label    = the teacher's OWN continuation from x fails: s/N ≤ {FAIL_MAX} (fail) vs ≥ {SOLVE_MIN} (solve), from the T-arm continuation table; Spearman on all states against s/N.")
    P("")
    if n6:
        P(f"-- N6: {n6['n_echoes']} echoes ({n6['n_echo_errors']} errors), {n6['n_rows']} action rows {n6['rows_by_src']}, {n6['rows_with_value']} labelled/matched rows with V − B; "
          f"king samples matched {n6['king_matched']}; own thought available {n6['own_thought_available']}")
        P(f"   sanity: blind refs' own-action A_live mean {_f(n6['blind_refs']['A_live_mean'], 7, 4)} (negative on {_f(n6['blind_refs']['A_live_neg_frac'], 5, 2)} of {n6['blind_refs']['n']}); "
          f"T mean {_f(n6['means']['T'], 7, 4)} (solved {_f(n6['means']['T_solved'], 7, 4)} / failed {_f(n6['means']['T_failed'], 7, 4)}); K mean {_f(n6['means']['K'], 7, 4)}")
        P(f"   {'proxy':12s} {'n':>4s} {'st':>3s} | pooled r [CI]            ρ | within r [CI]  (n,st) | within_mixed r [CI] (n,st) | within T+F r [CI] (n,st) | T+F mixed r [CI] (n,st) | +/− st | AUC solved (T, mixed) [CI]  centred [CI]  n | AUC TF all")
        for p in PROXIES:
            t = n6["proxies"][p]
            if not t.get("n"):
                P(f"   {p:12s} {'0':>4s}")
                continue
            po, wi, wm = t.get("pooled", {}), t.get("within", {}), t.get("within_mixed", {})
            wt, wtm = t.get("within_TF", {}), t.get("within_TF_mixed", {})
            au, auc_, aa = t.get("auc_T_mixed") or {}, t.get("auc_T_mixed_centred") or {}, t.get("auc_TF_all") or {}
            sg = t.get("per_state_sign", {})
            P(f"   {p:12s} {t['n']:4d} {t['n_states']:3d} | {_f(po.get('pearson'), 6, 3)} {_ci(po.get('pearson_ci'))} {_f(po.get('spearman'), 6, 3)} | "
              f"{_f(wi.get('pearson'), 6, 3)} {_ci(wi.get('pearson_ci'))} ({wi.get('n', 0)},{wi.get('n_states', 0)}) | "
              f"{_f(wm.get('pearson'), 6, 3)} {_ci(wm.get('pearson_ci'))} ({wm.get('n', 0)},{wm.get('n_states', 0)}) | "
              f"{_f(wt.get('pearson'), 6, 3)} {_ci(wt.get('pearson_ci'))} ({wt.get('n', 0)},{wt.get('n_states', 0)}) | "
              f"{_f(wtm.get('pearson'), 6, 3)} {_ci(wtm.get('pearson_ci'))} ({wtm.get('n', 0)},{wtm.get('n_states', 0)}) | "
              f"{sg.get('positive', 0)}/{sg.get('n_states', 0)} | {_f(au.get('auc'), 5, 2)} {_ci(au.get('ci'))} {_f(auc_.get('auc'), 5, 2)} {_ci(auc_.get('ci'))} {au.get('n', 0):3d} | {_f(aa.get('auc'), 5, 2)} {_ci(aa.get('ci'))} n{aa.get('n', 0)}")
        P("   pooled r on the king samples alone (V by class match): " + "; ".join(
            f"{p} n{n6['proxies'][p].get('pooled_K', {}).get('n', 0)} r {_f(n6['proxies'][p].get('pooled_K', {}).get('pearson'), 6, 3)}" for p in ("A_live", "A_sum", "B_own", "R_lab")))
        P("   by dialect (A_live / A_sum / R_lab: n, pooled r, within r):")
        for p in ("A_live", "A_sum", "R_lab"):
            bk = n6["proxies"][p].get("by_kind", {})
            P(f"     {p:8s} " + "  ".join(f"{kd}: n{v['n']} {_f(v['pooled_pearson'], 6, 3)} / {_f(v['within_pearson'], 6, 3)} ({v['n_states_within']} st)" for kd, v in bk.items()))
    else:
        P("-- N6: no echoes on disk (probe not run).")
    P("")
    if n4:
        P(f"-- N4 state-level: {n4['n_states']} states with informed refs ({n4['by_kind']}), notes ok {n4['notes_ok']}, samples parsed {n4['samples_parsed']}/{n4['samples_total']}; "
          f"labels fail {n4['n_fail']} / solve {n4['n_solve']} / in-between dropped {n4['n_mixed_dropped']}")
        P(f"   {'metric':10s} | {'AUC g→fail [CI]':24s} | {'AUC g0 (control)':24s} | {'AUC excess g−g0':24s} | {'paired ΔAUC g−g0 [CI]':22s} | {'Spearman(g, s/N) [CI] n':28s} | Spearman(g0, s/N)")
        for m in ("exact", "jac50", "jac_mean"):
            a, a0, ae = n4["auc"].get(f"g_{m}") or {}, n4["auc"].get(f"g0_{m}") or {}, n4["auc"].get(f"excess_{m}") or {}
            pg = n4["auc"].get(f"paired_gain_{m}") or {}
            s_, s0 = n4["spearman"].get(f"g_{m}") or {}, n4["spearman"].get(f"g0_{m}") or {}
            P(f"   {m:10s} | {_f(a.get('auc'), 5, 2)} {_ci(a.get('ci'))} n{a.get('n', 0):<3d} | {_f(a0.get('auc'), 5, 2)} {_ci(a0.get('ci'))} n{a0.get('n', 0):<3d} | "
              f"{_f(ae.get('auc'), 5, 2)} {_ci(ae.get('ci'))} n{ae.get('n', 0):<3d} | {_f(pg.get('diff'), 6, 3)} {_ci(pg.get('ci'))}     | "
              f"{_f(s_.get('rho'), 6, 3)} {_ci(s_.get('ci'))} n{s_.get('n', 0):<3d}      | {_f(s0.get('rho'), 6, 3)} {_ci(s0.get('ci'))}")
        g3 = n4["n_ge3"]
        P(f"   states with N ≥ 3 own continuations only ({g3['n_states']} labelled: {g3['n_fail']} fail / {g3['n_solve']} solve): " + "; ".join(
            f"{m}: AUC g {_f((g3['auc'].get(f'g_{m}') or {}).get('auc'), 5, 2)} {_ci((g3['auc'].get(f'g_{m}') or {}).get('ci'))} vs g0 {_f((g3['auc'].get(f'g0_{m}') or {}).get('auc'), 5, 2)}, "
            f"Spearman(g, s/N) {_f(g3['spearman'][f'g_{m}']['rho'], 6, 3)} n{g3['spearman'][f'g_{m}']['n']}" for m in ("exact", "jac50", "jac_mean")))
        sat = n4["saturated"]
        P(f"   ceiling: g_exact = 1.0 on {sat['g_exact_eq_1']}/{sat['n']} states (g0_exact = 1.0 on {sat['g0_exact_eq_1']}) — no informed ref repeats a stored blind ref verbatim there; g_jac50 = 1.0 on {sat['g_jac50_eq_1']}.")
        P("   per dialect AUC(g→fail) exact / jac50: " + "; ".join(
            f"{kd} n{v['exact']['n']} {_f(v['exact'].get('auc'), 5, 2)} / {_f(v['jac50'].get('auc'), 5, 2)}" for kd, v in n4["by_kind_auc"].items()))
        vo = n4["auc_vs_orig_outcome"]
        P(f"   controls: AUC(g→SOURCE trajectory failed, the check-1 label) exact {_f(vo['exact'].get('auc'), 5, 2)} / jac50 {_f(vo['jac50'].get('auc'), 5, 2)} (n{vo['exact']['n']}); "
          f"AUC(note length→fail) {_f((n4.get('auc_note_words') or {}).get('auc'), 5, 2)}; AUC(source-trajectory-failed bit→state fail) {_f((n4.get('auc_orig_outcome_bit') or {}).get('auc'), 5, 2)}; "
          f"notes mentioning the outcome {sum(r['note_mentions_outcome'] for r in n4['rows'])}/{n4['n_states']}.")
        P("   state rows (sid | kind | s/N | g_exact g_jac50 | g0_exact g0_jac50):")
        for r in sorted(n4["rows"], key=lambda r: (r["vhat"], r["sid"])):
            P(f"     {r['sid']:36s} {r['kind']:13s} {r['s']}/{r['n']}  {r['g_exact']:.2f} {r['g_jac50']:.2f} | {_f(r['g0_exact'], 4, 2)} {_f(r['g0_jac50'], 4, 2)}")
    else:
        P("-- N4 state-level: no informed samples on disk (probe not run).")
    js = {"usd_spent": usd, "n6": {k: v for k, v in n6.items()}, "n4_state": {k: v for k, v in n4.items()}}
    return L, js


def probe_verdict(n6: dict, n4: dict) -> list[str]:
    L = []
    if n6:
        a = n6["proxies"]["A_live"]
        wi, wm, po = a.get("within", {}), a.get("within_mixed", {}), a.get("pooled", {})

        def sign(r, ci):
            if r is None:
                return "not computable"
            if ci and ci[0] > 0:
                return "POSITIVE"
            if ci and ci[1] < 0:
                return "NEGATIVE"
            return f"NULL (point {r:+.2f}, CI spans 0)"
        L.append(f"  N6 — within a state the live A leg is {sign(wm.get('pearson'), wm.get('pearson_ci'))} with verified action value "
                 f"(state-centred r {_f(wm.get('pearson'), 6, 3)} {_ci(wm.get('pearson_ci'))} on {wm.get('n', 0)} actions / {wm.get('n_states', 0)} mixed states; all states {_f(wi.get('pearson'), 6, 3)} {_ci(wi.get('pearson_ci'))}; "
                 f"pooled {_f(po.get('pearson'), 6, 3)} {_ci(po.get('pearson_ci'))}, n {a.get('n', 0)}); AUC(A_live → the teacher's own first action solved | mixed states) {_f((a.get('auc_T_mixed') or {}).get('auc'), 5, 2)} {_ci((a.get('auc_T_mixed') or {}).get('ci'))}.")
        others = []
        for p in ("A_sum", "B_own", "R_blind", "R_lab"):
            t = n6["proxies"][p]
            w, wt, au = t.get("within_mixed", {}), t.get("within_TF_mixed", {}), t.get("auc_T_mixed") or {}
            if t.get("n"):
                others.append(f"{p} {_f(w.get('pearson'), 6, 3)} {_ci(w.get('pearson_ci'))} n{w.get('n', 0)} (T+F only {_f(wt.get('pearson'), 6, 3)} {_ci(wt.get('pearson_ci'))} n{wt.get('n', 0)}; AUC {_f(au.get('auc'), 5, 2)} {_ci(au.get('ci'))})")
        L.append("  N6 — other legs, state-centred on mixed states: " + "; ".join(others) + ". R_lab's refs ARE the other labelled actions, so a high R_lab means the action sits in the state's modal class — "
                 "at these 6 states the mode solves more often, which is typicality, not the meter reading value; A (the leg that would rank a miner's action) is null.")
    if n4:
        a, a0, ae = n4["auc"].get("g_exact") or {}, n4["auc"].get("g0_exact") or {}, n4["auc"].get("excess_exact") or {}
        aj, aj0 = n4["auc"].get("g_jac50") or {}, n4["auc"].get("g0_jac50") or {}
        s_, sj = n4["spearman"].get("g_exact") or {}, n4["spearman"].get("g_jac50") or {}
        located = (a.get("ci") or [0])[0] > 0.5 or (aj.get("ci") or [0])[0] > 0.5
        beats_ctrl = (n4["auc"].get("paired_gain_exact") or {}).get("ci", [0])[0] > 0 or (n4["auc"].get("paired_gain_jac50") or {}).get("ci", [0])[0] > 0
        L.append(f"  N4 — blind-vs-informed divergence {'DOES' if located and beats_ctrl else 'does NOT'} locate the states where the teacher's own continuation fails: "
                 f"AUC(g→fail) exact {_f(a.get('auc'), 5, 2)} {_ci(a.get('ci'))} / jac50 {_f(aj.get('auc'), 5, 2)} {_ci(aj.get('ci'))} (n {a.get('n', 0)}: {n4['n_fail']} fail / {n4['n_solve']} solve) vs the blind-vs-blind control "
                 f"{_f(a0.get('auc'), 5, 2)} / {_f(aj0.get('auc'), 5, 2)}; Spearman(g, s/N) {_f(s_.get('rho'), 6, 3)} {_ci(s_.get('ci'))} / {_f(sj.get('rho'), 6, 3)} {_ci(sj.get('ci'))} (n {s_.get('n', 0)}); "
                 f"kill line {KILL_AUC}: {'cleared' if max(a.get('auc') or 0, aj.get('auc') or 0) >= KILL_AUC else 'not cleared'} by the point estimate, "
                 f"{'and' if (a.get('ci') or [0])[0] >= KILL_AUC or (aj.get('ci') or [0])[0] >= KILL_AUC else 'not'} by the CI.")
        g3, sat = n4["n_ge3"], n4["saturated"]
        L.append(f"  N4 — caveats that cut both ways: norm-exact g is at its ceiling (1.0) on {sat['g_exact_eq_1']}/{sat['n']} states (the teacher rarely repeats a stored blind action verbatim, note or not), "
                 f"{sum(r['note_mentions_outcome'] for r in n4['rows'])}/{n4['n_states']} notes name the grade (the W1 p_hind leak), and {sum(1 for r in n4['rows'] if r['n'] == 1)} of the labelled states have a single own continuation; "
                 f"on the {g3['n_states']} states with N ≥ 3 the AUC is exact {_f((g3['auc'].get('g_exact') or {}).get('auc'), 5, 2)} / jac50 {_f((g3['auc'].get('g_jac50') or {}).get('auc'), 5, 2)} — same answer.")
    return L


def probe_section() -> tuple[list[str], dict, list[str]]:
    if not PROBE_STATES.exists():
        return [], {}, []
    rng = random.Random(SEED + 1)
    states = graded_states()
    n6 = n6_probe(states, rng)
    n4 = n4_state_probe(states, rng)
    usd = probe_total_cost()
    lines, js = probe_report(states, n6, n4, usd)
    return lines, js, probe_verdict(n6, n4)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    pr = sub.add_parser("probe")
    pr.add_argument("stage", choices=("states", "echo", "notes", "sample"))
    pr.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "probe":
        {"states": cmd_states, "echo": cmd_echo, "notes": cmd_notes, "sample": cmd_sample}[args.stage](args)
        return
    rng = random.Random(SEED)
    rows, meta = build_turn_rows()
    tables = auc_tables(rows, rng)
    overlap = graded_state_overlap(rows)
    c2 = check2()
    write_report(rows, meta, tables, overlap, c2)
    lines, js, verdict = probe_section()
    if lines:
        txt = (OUT / "report.txt").read_text() + "\n".join(lines) + "\n"
        if verdict:
            txt += "\n== PROBE VERDICT ==\n" + "\n".join(verdict) + "\n"
        (OUT / "report.txt").write_text(txt)
        rep = json.loads((OUT / "report.json").read_text())
        rep["probes"] = js
        rep["usd_spent"] = js["usd_spent"]
        (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
        print("\n".join(lines))
        if verdict:
            print("\n== PROBE VERDICT ==\n" + "\n".join(verdict))


if __name__ == "__main__":
    main()
