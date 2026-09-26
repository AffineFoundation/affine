"""X1 — in-band selection of privileged refs (S3a) and the paraphrase guard
(S3b), on top of privileged_refs.py's stored p_fr data.

    python inband_select.py stored                 # X1.1 from stored samples/echoes/judge
    python inband_select.py sample [--turns 60]    # X1.2 guarded N=6 p_fr refs
    python inband_select.py echo
    python inband_select.py judge  [--turns 30]
    python inband_select.py report                 # X1.2 tables (+ re-runs X1.1)

Terms (one line each):
  in-band          |m_c(z) − mean_3 m_c(blind)| ≤ 2·σ_Mc(dialect) AND ≥10 content tokens
                   (live typ_c ≥ 0); blind band = the verdict's 3 stored teacher refs re-echoed
                   on the Engy stack, σ from the verdict's shadow sd_meter.
  change           the ref's action is norm-exact-different from all 3 blind refs AND its best
                   token-Jaccard to them is < 0.5 (the outcome probe's CONTESTED rule).
  judge SAME       glm-5.3-flash T=0 "same decision" (ref i vs blind ref i mod 3).
  leak             a ≥30-char substring of the note appears in z or y (guarded refs so
                   flagged are rejected from every reference set).
  B                lpC(y|x,z) − lpC(y|x,∅) per byte; licence at ≥ 0.02.
  R_priv_S(z)      centered tempered LME (τ .03) of a_i = lpC(y_i|x,z) − lpC(y_i|x,∅) over the
                   reference set S; external thoughts (blind-own, king) 2-ref LOO-matched =
                   mean over 2-subsets of S; own thoughts = ref j's thought over S \\ {j}
                   (needs |S| ≥ 3).
  hz / d/σ         paired z over turns of the difference / mean difference in σ_R(dialect) units.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import random
import re
import statistics as st
import sys
import time
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    REPO, TEACHER_ENGY, Engy, TeacherEcho, agree, clme, exact, jaccard, norm_action,
    read_jsonl, append_jsonl, write_jsonl,
)
from privileged_refs import (  # noqa: E402
    ECHOES, JUDGE, JUDGE_MODEL, JUDGE_SYSTEM, K, NOTE_HEAD, NOTES, REF_MAX_TOKENS,
    REF_TEMPERATURE, SAMPLES, TAU, TURNS, leaks, parse_reply, read_echoes, with_note,
)
from evalsrv.sdmeter import content_stats  # noqa: E402

OUT = REPO / "research" / "results" / "frontier_arbiter" / "inband"
G_SAMPLES = OUT / "guard_samples.jsonl"
G_ECHOES = OUT / "guard_echoes.jsonl"
G_JUDGE = OUT / "guard_judge.jsonl"
G_TURNS = OUT / "guard_turns.json"
COST = OUT / "cost.jsonl"

GUARD = ("Do not quote or restate this note; reason in your own words in at most 150 words, "
         "then act.")
N_GUARD = 6
CHANGE_JAC = 0.5
QUOTA = {"bash": 24, "tool_call": 24, "terminus_json": 12}


# ------------------------------------------------------------------ helpers
def log_cost(stage: str, engy: Engy, note: str = "") -> None:
    append_jsonl(COST, {"at": time.time(), "stage": stage, "cost_usd": engy.cost_usd,
                        "usage": engy.usage, "note": note})
    print(f"  [$] {stage}: this run ${engy.cost_usd:.3f} {note}", flush=True)


def total_cost() -> float:
    """Each run appends cumulative cost_usd per checkpoint; a drop = new run."""
    tot = 0.0
    by_stage: dict[str, list[float]] = collections.defaultdict(list)
    for r in read_jsonl(COST):
        by_stage[r["stage"]].append(r["cost_usd"])
    for vals in by_stage.values():
        run_max = 0.0
        for c in vals:
            if c < run_max:
                tot += run_max
                run_max = 0.0
            run_max = max(run_max, c)
        tot += run_max
    return tot


def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return st.mean(v) if v else None


def _p50(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def _paired(diffs: list[float]) -> dict:
    d = [x for x in diffs if x is not None and math.isfinite(x)]
    if len(d) < 3:
        return {"n": len(d), "mean": _mean(d), "se": None, "z": None}
    se = st.stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "mean": st.mean(d), "se": se, "z": (st.mean(d) / se) if se > 0 else None}


def _rate(flags):
    f = [x for x in flags if x is not None]
    return (sum(1.0 for x in f if x) / len(f)) if f else None


def mc_of(ex: dict | None, eu: dict | None) -> tuple[float | None, int | None]:
    if not ex or not eu:
        return None, None
    cs = content_stats([tuple(x) for x in ex["tokens"]], [tuple(x) for x in eu["tokens"]], 1.0)
    return cs["mc"], cs["n_content"]


def R_external(a_by_ref: dict[int, float], S: list[int]) -> float | None:
    """2-ref LOO-matched R of an external thought over reference set S."""
    if len(S) < 2:
        return None
    return st.mean(clme([a_by_ref[i] for i in sub], TAU) for sub in combinations(sorted(S), 2))


def R_own(a_cross: dict[tuple[int, int], float], S: list[int]) -> float | None:
    """mean over j in S of clme over i in S\\{j} of a_cross[(i, j)] = lpC(y_i|x,z_j) − lpC(y_i|x,∅).
    Matched to the 2-ref external estimator: with |S| = 3 each own value uses 2 refs; with
    |S| > 3 the 2-subsets of S\\{j} are averaged."""
    if len(S) < 3:
        return None
    vals = []
    for j in S:
        others = [i for i in S if i != j]
        subs = list(combinations(sorted(others), 2))
        vals.append(st.mean(clme([a_cross[(i, j)] for i in sub], TAU) for sub in subs))
    return st.mean(vals)


def binom_ge(p: float, n: int, m: int) -> float:
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(m, n + 1))


# ------------------------------------------------------------------ per-ref rows
def turn_context(t: dict, echoes: dict) -> dict:
    tid = t["turn_id"]
    blind = t["refs"]
    mcb = [mc_of(echoes.get(f"{tid}|th|blind|{i}"), echoes.get(f"{tid}|un|blind|{i}"))[0] for i in range(K)]
    mu = st.mean(mcb) if all(m is not None for m in mcb) else None
    return {"tid": tid, "kind": t["dialect"], "blind": blind, "blind_ys": [r["y"] for r in blind],
            "king_z": t["king"]["pairs"][0]["z_a"], "sig": t.get("sigma_mc"), "sig_r": t.get("sigma_r"),
            "mu_b": mu, "mc_blind": mcb, "outcome": t["down"]["outcome"]}


def ref_row(ctx: dict, s: dict, note: str, echoes: dict, prefix_key: str, judge_same, ctrl_ys: list[str]) -> dict:
    """One parsed ref → flags. prefix_key = "{tid}|..|p_fr|{i}" style key root for echoes."""
    tid, kind, i = ctx["tid"], ctx["kind"], s["i"]
    mc, nc = mc_of(echoes.get(f"{tid}|th|{prefix_key}|{i}"), echoes.get(f"{tid}|un|{prefix_key}|{i}"))
    z = ((mc - ctx["mu_b"]) / ctx["sig"]) if (mc is not None and ctx["mu_b"] is not None and ctx["sig"]) else None
    emp = echoes.get(f"{tid}|emp|{prefix_key}|{i}")
    own = echoes.get(f"{tid}|own|{prefix_key}|{i}")
    B = (own["lp_per_byte"] - emp["lp_per_byte"]) if (emp and own) else None
    y = s["y"]
    ex_b = exact(y, ctx["blind_ys"], kind)
    jac_b = agree(y, ctx["blind_ys"])
    row = {"turn_id": tid, "dialect": kind, "outcome": ctx["outcome"], "i": i,
           "mc": mc, "n_content": nc, "z_live": z,
           "in_band": (z is not None and abs(z) <= 2.0 and (nc or 0) >= 10),
           "content_floor": (nc is not None and nc < 10),
           "exact_blind": ex_b, "jac_blind": jac_b, "change": (not ex_b) and jac_b < CHANGE_JAC,
           "exact_ctrl": exact(y, ctrl_ys, kind) if ctrl_ys else None,
           "jac_ctrl": agree(y, ctrl_ys) if ctrl_ys else None,
           "judge_same": judge_same,
           "leak_z": leaks(note, s["z"]), "leak_y": leaks(note, y),
           "len_z": len(s["z"]), "len_y": len(y), "cap": s.get("finish") == "length",
           "B": B, "B_pass": (B is not None and B >= 0.02),
           "visible": ("</think>" in s["z"])}
    row["leak"] = row["leak_z"] or row["leak_y"]
    a_blind = {}
    for b in range(K):
        e = echoes.get(f"{tid}|x|{prefix_key}|{i}|blind|{b}")
        if e and emp:
            a_blind[b] = e["lp_per_byte"] - emp["lp_per_byte"]
    row["a_blind"] = a_blind
    e = echoes.get(f"{tid}|x|{prefix_key}|{i}|king|0")
    row["a_king"] = (e["lp_per_byte"] - emp["lp_per_byte"]) if (e and emp) else None
    return row


def headroom_for_set(rows: list[dict], S_ids: list[int], cross: dict[tuple[int, int], float]) -> dict:
    """R_priv over reference set S (ids into rows by ref index)."""
    by_i = {r["i"]: r for r in rows}
    out = {"n_S": len(S_ids)}
    if len(S_ids) < 2:
        return out
    bl = []
    for b in range(K):
        a = {i: by_i[i]["a_blind"].get(b) for i in S_ids}
        if all(v is not None for v in a.values()):
            bl.append(R_external(a, S_ids))
    out["R_bown"] = st.mean(bl) if len(bl) == K else None
    a = {i: by_i[i]["a_king"] for i in S_ids}
    out["R_king"] = R_external(a, S_ids) if all(v is not None for v in a.values()) else None
    if len(S_ids) >= 3 and all((i, j) in cross for i in S_ids for j in S_ids if i != j):
        out["R_own"] = R_own(cross, S_ids)
    else:
        out["R_own"] = None
    return out


def aggregate_refs(rows: list[dict]) -> dict:
    return {"n": len(rows),
            "change": _rate([r["change"] for r in rows]),
            "exact_blind": _rate([r["exact_blind"] for r in rows]),
            "jac_blind": _mean([r["jac_blind"] for r in rows]),
            "exact_ctrl": _rate([r["exact_ctrl"] for r in rows]),
            "jac_ctrl": _mean([r["jac_ctrl"] for r in rows]),
            "judge_same": _rate([r["judge_same"] for r in rows]),
            "judge_n": sum(1 for r in rows if r["judge_same"] is not None),
            "leak": _rate([r["leak"] for r in rows]),
            "leak_z": _rate([r["leak_z"] for r in rows]),
            "len_z_p50": _p50([r["len_z"] for r in rows]),
            "len_z_mean": _mean([r["len_z"] for r in rows]),
            "len_y_p50": _p50([r["len_y"] for r in rows]),
            "B_mean": _mean([r["B"] for r in rows]),
            "B_pass": _rate([r["B_pass"] for r in rows if r["B"] is not None]),
            "cap": _rate([r["cap"] for r in rows]),
            "abs_z_p50": _p50([abs(r["z_live"]) for r in rows if r["z_live"] is not None]),
            "z_mean": _mean([r["z_live"] for r in rows]),
            "visible": _rate([r["visible"] for r in rows])}


def headroom_table(per_turn: list[dict], key: str) -> dict:
    """per_turn[i][key] = headroom_for_set output; paired diffs over turns."""
    hs = [(t, t[key]) for t in per_turn if key in t]
    out = {"n_turns_ge2": sum(1 for _, h in hs if h.get("n_S", 0) >= 2),
           "n_turns_ge3": sum(1 for _, h in hs if h.get("n_S", 0) >= 3)}
    own_b = [(h["R_own"] - h["R_bown"], t["sig_r"]) for t, h in hs if h.get("R_own") is not None and h.get("R_bown") is not None]
    own_k = [(h["R_own"] - h["R_king"], t["sig_r"]) for t, h in hs if h.get("R_own") is not None and h.get("R_king") is not None]
    b_k = [(h["R_bown"] - h["R_king"], t["sig_r"]) for t, h in hs if h.get("R_bown") is not None and h.get("R_king") is not None]
    out["R_own"] = _mean([h["R_own"] for _, h in hs if h.get("R_own") is not None])
    out["R_bown_ge3"] = _mean([h["R_bown"] for _, h in hs if h.get("R_own") is not None])
    out["R_king_ge3"] = _mean([h["R_king"] for _, h in hs if h.get("R_own") is not None])
    out["R_bown"] = _mean([h["R_bown"] for _, h in hs if h.get("R_bown") is not None])
    out["R_king"] = _mean([h["R_king"] for _, h in hs if h.get("R_king") is not None])
    out["hz_own_bown"] = _paired([d for d, _ in own_b])
    out["hz_own_king"] = _paired([d for d, _ in own_k])
    out["hz_bown_king"] = _paired([d for d, _ in b_k])
    out["eff_own_bown"] = _mean([d / s for d, s in own_b if s])
    out["eff_own_king"] = _mean([d / s for d, s in own_k if s])
    out["eff_bown_king"] = _mean([d / s for d, s in b_k if s])
    return out


# ------------------------------------------------------------------ X1.1 stored
def stored_analysis() -> dict:
    turns = read_jsonl(TURNS)
    samples = [s for s in read_jsonl(SAMPLES) if "error" not in s]
    echoes = {r["key"]: r for r in read_echoes() if "error" not in r}
    notes = {n["turn_id"]: n["note"] for n in read_jsonl(NOTES) if n.get("note")}
    judge = {(j["turn_id"], j["variant"], j["pair"]): j["same"] for j in read_jsonl(JUDGE) if j.get("same") is not None}
    by_turn = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in samples:
        if s.get("parsed"):
            by_turn[s["turn_id"]][s["variant"]].append(s)
    ref_rows: dict[str, list[dict]] = {"p_fr": [], "p_none": []}
    per_turn: list[dict] = []
    for t in turns:
        ctx = turn_context(t, echoes)
        tid = ctx["tid"]
        note = notes.get(tid, "")
        ctrl_ys = [s["y"] for s in by_turn[tid].get("p_none", [])]
        pt = {"turn_id": tid, "dialect": ctx["kind"], "outcome": ctx["outcome"], "sig_r": ctx["sig_r"]}
        for v in ("p_fr", "p_none"):
            ss = sorted(by_turn[tid].get(v, []), key=lambda s: s["i"])
            rows = []
            for s in ss:
                js = judge.get((tid, v, s["i"])) if s["i"] in (0, 1) else None
                rows.append(ref_row(ctx, s, note if v == "p_fr" else "", echoes, v, js,
                                    ctrl_ys if v == "p_fr" else []))
            for r in rows:
                r["variant"] = v
            ref_rows[v].extend(rows)
            if v != "p_fr":
                pt["p_none_inband"] = sum(1 for r in rows if r["in_band"])
                pt["p_none_parsed"] = len(rows)
                continue
            cross = {}
            for r in rows:
                for r2 in rows:
                    if r2["i"] == r["i"]:
                        continue
                    e = echoes.get(f"{tid}|x|p_fr|{r['i']}|p_fr|{r2['i']}")
                    emp = echoes.get(f"{tid}|emp|p_fr|{r['i']}")
                    if e and emp:
                        cross[(r["i"], r2["i"])] = e["lp_per_byte"] - emp["lp_per_byte"]
            ids_all = [r["i"] for r in rows]
            S_in = [r["i"] for r in rows if r["in_band"]]
            S_in_clean = [r["i"] for r in rows if r["in_band"] and not r["leak"]]
            S_out = [r["i"] for r in rows if not r["in_band"]]
            pt.update({"n_parsed": len(rows), "n_inband": len(S_in), "n_inband_clean": len(S_in_clean),
                       "n_leak": sum(1 for r in rows if r["leak"]),
                       "h_all": headroom_for_set(rows, ids_all, cross),
                       "h_in": headroom_for_set(rows, S_in, cross),
                       "h_in_clean": headroom_for_set(rows, S_in_clean, cross),
                       "h_out": headroom_for_set(rows, S_out, cross)})
        per_turn.append(pt)
    write_jsonl(OUT / "stored_ref_rows.jsonl", ref_rows["p_fr"] + ref_rows["p_none"])
    write_jsonl(OUT / "stored_turn_rows.jsonl", per_turn)
    rep: dict = {"n_turns": len(turns)}
    fr = ref_rows["p_fr"]
    rep["refs"] = {"p_fr_in": aggregate_refs([r for r in fr if r["in_band"]]),
                   "p_fr_out": aggregate_refs([r for r in fr if not r["in_band"]]),
                   "p_fr_all": aggregate_refs(fr),
                   "p_none_in": aggregate_refs([r for r in ref_rows["p_none"] if r["in_band"]]),
                   "p_none_out": aggregate_refs([r for r in ref_rows["p_none"] if not r["in_band"]]),
                   "p_none_all": aggregate_refs(ref_rows["p_none"])}
    rep["refs_by_dialect"] = {k: {"p_fr_in": aggregate_refs([r for r in fr if r["in_band"] and r["dialect"] == k]),
                                  "p_fr_out": aggregate_refs([r for r in fr if not r["in_band"] and r["dialect"] == k])}
                              for k in ("bash", "tool_call", "terminus_json")}
    # in-band vs change: 2x2 + Fisher-ish odds
    a = sum(1 for r in fr if r["in_band"] and r["change"])
    b = sum(1 for r in fr if r["in_band"] and not r["change"])
    c = sum(1 for r in fr if not r["in_band"] and r["change"])
    d = sum(1 for r in fr if not r["in_band"] and not r["change"])
    rep["inband_x_change"] = {"in_change": a, "in_same": b, "out_change": c, "out_same": d,
                              "odds_ratio": ((a * d) / (b * c)) if b * c else None}
    # z of the difference in change rate (two-proportion)
    p1, p2 = a / max(1, a + b), c / max(1, c + d)
    pp = (a + c) / max(1, a + b + c + d)
    se = math.sqrt(pp * (1 - pp) * (1 / max(1, a + b) + 1 / max(1, c + d))) if 0 < pp < 1 else None
    rep["inband_x_change"]["z_diff"] = ((p1 - p2) / se) if se else None
    # z_live vs change: mean |z| of changed vs same refs
    rep["absz_by_change"] = {"change": _p50([abs(r["z_live"]) for r in fr if r["change"] and r["z_live"] is not None]),
                             "same": _p50([abs(r["z_live"]) for r in fr if not r["change"] and r["z_live"] is not None])}
    rep["headroom"] = {k: headroom_table(per_turn, k) for k in ("h_all", "h_in", "h_in_clean", "h_out")}
    rep["headroom_by_dialect"] = {kd: {k: headroom_table([p for p in per_turn if p["dialect"] == kd], k) for k in ("h_all", "h_in")}
                                  for kd in ("bash", "tool_call", "terminus_json")}
    # coverage at k=3 and projected N=6
    cov = {"k3_ge1": _rate([p.get("n_inband", 0) >= 1 for p in per_turn]),
           "k3_ge2": _rate([p.get("n_inband", 0) >= 2 for p in per_turn]),
           "k3_ge3": _rate([p.get("n_inband", 0) >= 3 for p in per_turn]),
           "k3_ge1_clean": _rate([p.get("n_inband_clean", 0) >= 1 for p in per_turn]),
           "k3_ge2_clean": _rate([p.get("n_inband_clean", 0) >= 2 for p in per_turn]),
           "k3_ge3_clean": _rate([p.get("n_inband_clean", 0) >= 3 for p in per_turn])}
    # per-turn in-band probability = in-band / sampled(3) (unparsed count as not in band)
    pts = [p.get("n_inband", 0) / K for p in per_turn]
    pooled = st.mean(pts)
    for m in (1, 2, 3):
        cov[f"n6_ge{m}_perturn"] = st.mean(binom_ge(p, N_GUARD, m) for p in pts)
        cov[f"n6_ge{m}_pooled"] = binom_ge(pooled, N_GUARD, m)
    # shrunk per-turn rates (beta(1,1) prior) — the 3-sample plug-in is noisy
    pts_s = [(p.get("n_inband", 0) + pooled * 2) / (K + 2) for p in per_turn]
    for m in (1, 2, 3):
        cov[f"n6_ge{m}_shrunk"] = st.mean(binom_ge(p, N_GUARD, m) for p in pts_s)
    cov["per_ref_inband"] = pooled
    cov["p_none_k3_ge2"] = _rate([p.get("p_none_inband", 0) >= 2 for p in per_turn])
    cov["p_none_k3_ge3"] = _rate([p.get("p_none_inband", 0) >= 3 for p in per_turn])
    rep["coverage"] = cov
    return rep


# ------------------------------------------------------------------ X1.2 guarded resample
def pick_turns(turns: list[dict], n: int) -> list[str]:
    if G_TURNS.exists():
        return json.loads(G_TURNS.read_text())
    rng = random.Random(20260921)
    pool = collections.defaultdict(list)
    for t in turns:
        pool[t["dialect"]].append(t["turn_id"])
    picked = []
    for k, q in QUOTA.items():
        ids = pool[k]
        rng.shuffle(ids)
        picked.extend(ids[:q])
    picked = picked[:n]
    OUT.mkdir(parents=True, exist_ok=True)
    G_TURNS.write_text(json.dumps(picked))
    return picked


def guarded_note(note: str) -> str:
    return GUARD + "\n" + note


async def sample_guarded(turns: list[dict], engy: Engy, notes: dict[str, str]) -> None:
    done = {(r["turn_id"], r["i"]) for r in read_jsonl(G_SAMPLES) if "error" not in r}
    jobs = [(t, i) for t in turns for i in range(N_GUARD) if (t["turn_id"], i) not in done]
    print(f"guarded sampling: {len(jobs)} refs")

    async def one(t, i):
        p = guarded_note(notes[t["turn_id"]])
        msgs = with_note(t["prefix"], p)
        try:
            r = await engy.chat(TEACHER_ENGY, msgs, temperature=REF_TEMPERATURE, max_tokens=REF_MAX_TOKENS)
        except Exception as ex:  # noqa: BLE001
            append_jsonl(G_SAMPLES, {"turn_id": t["turn_id"], "i": i, "error": repr(ex)[:300]})
            return
        parsed = parse_reply(r, t["dialect"])
        note = notes[t["turn_id"]]
        append_jsonl(G_SAMPLES, {"turn_id": t["turn_id"], "i": i, "note": note, "guard": GUARD,
                                 "reasoning": r["reasoning"], "content": r["content"],
                                 "tool_calls": r["tool_calls"], "finish": r["finish"], "usage": r["usage"],
                                 "cost_usd": r["cost_usd"],
                                 **{k: parsed[k] for k in ("z", "y", "parsed", "kind_used", "think_closed", "repaired")},
                                 "leak_z": leaks(note, parsed["z"]), "leak_y": leaks(note, parsed["y"]),
                                 "leak_guard": leaks(GUARD, parsed["z"]) or leaks(GUARD, parsed["y"])})

    step = 48
    for s in range(0, len(jobs), step):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + step]])
        log_cost("sample", engy, f"{min(s + step, len(jobs))}/{len(jobs)}")


def cmd_sample(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    ids = set(pick_turns(turns, args.turns))
    sel = [t for t in turns if t["turn_id"] in ids]
    notes = {n["turn_id"]: n["note"] for n in read_jsonl(NOTES) if n.get("note")}
    engy = Engy(concurrency=args.concurrency)
    asyncio.run(sample_guarded(sel, engy, notes))


def guard_refs(tid: str, samples: list[dict]) -> list[dict]:
    return sorted([s for s in samples if s["turn_id"] == tid and s.get("parsed")], key=lambda s: s["i"])


def guard_echo_jobs(t: dict, refs: list[dict], done: set[str]) -> list[tuple]:
    tid = t["turn_id"]
    prefix = t["prefix"]
    blind = t["refs"]
    king_z = t["king"]["pairs"][0]["z_a"]
    jobs = []

    def add(key, kind, z, y=None):
        if key not in done:
            jobs.append((key, kind, prefix, z, y))
    V = "g_fr"
    clean = [r for r in refs if not (r["leak_z"] or r["leak_y"])]
    for r in refs:
        add(f"{tid}|th|{V}|{r['i']}", "thought", r["z"])
        add(f"{tid}|un|{V}|{r['i']}", "uncond", r["z"])
    for r in clean:
        add(f"{tid}|emp|{V}|{r['i']}", "action", "", r["y"])
        add(f"{tid}|own|{V}|{r['i']}", "action", r["z"], r["y"])
        for b in range(K):
            add(f"{tid}|x|{V}|{r['i']}|blind|{b}", "action", blind[b]["z"], r["y"])
        add(f"{tid}|x|{V}|{r['i']}|king|0", "action", king_z, r["y"])
        for r2 in clean:
            if r2["i"] != r["i"]:
                add(f"{tid}|x|{V}|{r['i']}|{V}|{r2['i']}", "action", r2["z"], r["y"])
    return jobs


def read_guard_echoes() -> list[dict]:
    out: list[dict] = []
    for p in sorted(OUT.glob("guard_echoes*.jsonl")):
        out.extend(read_jsonl(p))
    return out


async def echo_guarded(turns: list[dict], engy: Engy, shard: int = 0) -> None:
    te = TeacherEcho(engy)
    samples = [s for s in read_jsonl(G_SAMPLES) if "error" not in s]
    done = {r["key"] for r in read_guard_echoes() if "error" not in r}
    # parallel shards append to their own part file so lines never interleave
    out_path = G_ECHOES if shard == 0 else OUT / f"guard_echoes.part{shard}.jsonl"
    jobs = []
    for t in turns:
        jobs.extend(guard_echo_jobs(t, guard_refs(t["turn_id"], samples), done))
    print(f"guarded echo (shard {shard}): {len(jobs)} echoes")

    async def one(key, kind, prefix, z, y):
        try:
            if kind == "action":
                r = await te.lp_action(prefix, z, y)
            elif kind == "thought":
                r = await te.lp_thought(prefix, z, tokens=True)
            else:
                r = await te.lp_thought_uncond(z, tokens=True)
            append_jsonl(out_path, {"key": key, **r})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(out_path, {"key": key, "error": repr(ex)[:300]})

    # stream: the Engy semaphore bounds concurrency; no batch waits on a straggler
    n_done = 0
    tasks = [asyncio.ensure_future(one(*j)) for j in jobs]
    for fut in asyncio.as_completed(tasks):
        await fut
        n_done += 1
        if n_done % 96 == 0 or n_done == len(jobs):
            log_cost(f"echo{shard}", engy, f"{n_done}/{len(jobs)}")


def cmd_echo(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    ids = pick_turns(turns, 60)
    sel = [t for t in turns if t["turn_id"] in set(ids)]
    sel = sel[args.shard::args.nshards]
    engy = Engy(concurrency=args.concurrency)
    asyncio.run(echo_guarded(sel, engy, args.shard))


async def judge_guarded(turns: list[dict], engy: Engy) -> None:
    samples = [s for s in read_jsonl(G_SAMPLES) if "error" not in s]
    done = {(r["turn_id"], r["i"]) for r in read_jsonl(G_JUDGE) if r.get("same") is not None}
    jobs = []
    for t in turns:
        tail = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in t["prefix"][-3:])
        if len(tail) > 12_000:
            tail = tail[-12_000:]
        for r in guard_refs(t["turn_id"], samples):
            if (t["turn_id"], r["i"]) in done:
                continue
            b = t["refs"][r["i"] % K]["y"]
            jobs.append((t["turn_id"], r["i"], tail, b, r["y"]))
    print(f"guarded judge: {len(jobs)} comparisons")

    async def one(tid, i, tail, ya, yb):
        user = (f"Recent transcript context:\n{tail}\n\n--- Candidate action A ---\n{ya[:3000]}\n\n"
                f"--- Candidate action B ---\n{yb[:3000]}\n\nSAME or DIFFERENT?")
        try:
            r = await engy.chat(JUDGE_MODEL, [{"role": "system", "content": JUDGE_SYSTEM},
                                              {"role": "user", "content": user}], temperature=0.0, max_tokens=2000)
            ans = (r["content"] or "").strip().upper()
            if not ans.startswith(("SAME", "DIFF")):
                words = re.findall(r"\b(SAME|DIFFERENT)\b", (r["content"] or "") + " " + (r["reasoning"] or ""), re.I)
                ans = words[-1].upper() if words else ans
            same = ans.startswith("SAME") if ans.startswith(("SAME", "DIFF")) else None
            append_jsonl(G_JUDGE, {"turn_id": tid, "i": i, "same": same, "raw": ans[:40], "cost_usd": r["cost_usd"]})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(G_JUDGE, {"turn_id": tid, "i": i, "same": None, "error": repr(ex)[:200]})

    for s in range(0, len(jobs), 32):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + 32]])
        log_cost("judge", engy, f"{min(s + 32, len(jobs))}/{len(jobs)}")


def cmd_judge(args: argparse.Namespace) -> None:
    turns = read_jsonl(TURNS)
    ids = pick_turns(turns, 60)
    # 30-turn subsample: dialect-interleaved so every dialect is represented
    by_k = collections.defaultdict(list)
    for tid in ids:
        by_k[next(t["dialect"] for t in turns if t["turn_id"] == tid)].append(tid)
    sub = []
    while len(sub) < args.turns and any(by_k.values()):
        for k in ("bash", "tool_call", "terminus_json"):
            if by_k[k] and len(sub) < args.turns:
                sub.append(by_k[k].pop(0))
    (OUT / "guard_judge_turns.json").write_text(json.dumps(sub))
    sel = [t for t in turns if t["turn_id"] in set(sub)]
    engy = Engy(concurrency=16)
    asyncio.run(judge_guarded(sel, engy))


# ------------------------------------------------------------------ X1.2 analysis
def guarded_analysis() -> dict:
    turns = read_jsonl(TURNS)
    ids = pick_turns(turns, 60)
    sel = [t for t in turns if t["turn_id"] in set(ids)]
    samples_all = read_jsonl(G_SAMPLES)
    samples = [s for s in samples_all if "error" not in s]
    echoes = {r["key"]: r for r in read_echoes() if "error" not in r}
    echoes.update({r["key"]: r for r in read_guard_echoes() if "error" not in r})
    judge = {(j["turn_id"], j["i"]): j["same"] for j in read_jsonl(G_JUDGE) if j.get("same") is not None}
    notes = {n["turn_id"]: n["note"] for n in read_jsonl(NOTES) if n.get("note")}
    stored_samples = [s for s in read_jsonl(SAMPLES) if "error" not in s and s.get("parsed")]
    ref_rows: list[dict] = []
    per_turn: list[dict] = []
    raw = {"n_sampled": 0, "n_parsed": 0, "n_cap": 0, "n_leak": 0, "n_leak_z": 0, "n_leak_y": 0,
           "n_leak_guard": 0, "n_think_closed": 0, "n_text_fb": 0, "completion_tokens": 0, "prompt_tokens": 0}
    for t in sel:
        ctx = turn_context(t, echoes)
        tid = ctx["tid"]
        note = notes[tid]
        ss = sorted([s for s in samples if s["turn_id"] == tid], key=lambda s: s["i"])
        raw["n_sampled"] += len(ss)
        raw["n_parsed"] += sum(1 for s in ss if s["parsed"])
        raw["n_cap"] += sum(1 for s in ss if s["finish"] == "length")
        raw["n_think_closed"] += sum(1 for s in ss if s["think_closed"])
        raw["n_text_fb"] += sum(1 for s in ss if s.get("kind_used") == "text")
        raw["n_leak"] += sum(1 for s in ss if s["parsed"] and (s["leak_z"] or s["leak_y"]))
        raw["n_leak_z"] += sum(1 for s in ss if s["parsed"] and s["leak_z"])
        raw["n_leak_y"] += sum(1 for s in ss if s["parsed"] and s["leak_y"])
        raw["n_leak_guard"] += sum(1 for s in ss if s["parsed"] and s.get("leak_guard"))
        raw["completion_tokens"] += sum((s.get("usage") or {}).get("completion_tokens") or 0 for s in ss)
        raw["prompt_tokens"] += sum((s.get("usage") or {}).get("prompt_tokens") or 0 for s in ss)
        refs = [s for s in ss if s["parsed"]]
        ctrl_ys = [s["y"] for s in stored_samples if s["turn_id"] == tid and s["variant"] == "p_none"]
        rows = [ref_row(ctx, s, note, echoes, "g_fr", judge.get((tid, s["i"])), ctrl_ys) for s in refs]
        for r in rows:
            r["variant"] = "g_fr"
        ref_rows.extend(rows)
        cross = {}
        for r in rows:
            for r2 in rows:
                if r2["i"] == r["i"]:
                    continue
                e = echoes.get(f"{tid}|x|g_fr|{r['i']}|g_fr|{r2['i']}")
                emp = echoes.get(f"{tid}|emp|g_fr|{r['i']}")
                if e and emp:
                    cross[(r["i"], r2["i"])] = e["lp_per_byte"] - emp["lp_per_byte"]
        clean = [r["i"] for r in rows if not r["leak"]]
        S_in = [r["i"] for r in rows if r["in_band"] and not r["leak"]]
        S_out = [r["i"] for r in rows if not r["in_band"] and not r["leak"]]
        # the stored (unguarded) p_fr on the same turn, for the paired comparison
        pf = sorted([s for s in stored_samples if s["turn_id"] == tid and s["variant"] == "p_fr"], key=lambda s: s["i"])
        pf_rows = [ref_row(ctx, s, note, echoes, "p_fr", None, ctrl_ys) for s in pf]
        pt = {"turn_id": tid, "dialect": ctx["kind"], "outcome": ctx["outcome"], "sig_r": ctx["sig_r"],
              "n_parsed": len(rows), "n_clean": len(clean), "n_inband": len(S_in),
              "n_inband_any": sum(1 for r in rows if r["in_band"]),
              "h_clean": headroom_for_set(rows, clean, cross),
              "h_in": headroom_for_set(rows, S_in, cross),
              "h_in3": headroom_for_set(rows, S_in[:3], cross),          # k=3-matched: first 3 in-band
              "h_out": headroom_for_set(rows, S_out, cross),
              "h_out3": headroom_for_set(rows, S_out[:3], cross),
              "pfr_inband": sum(1 for r in pf_rows if r["in_band"]), "pfr_parsed": len(pf_rows),
              "pfr_leak": sum(1 for r in pf_rows if r["leak"]),
              "pfr_change": _rate([r["change"] for r in pf_rows]),
              "g_change": _rate([r["change"] for r in rows])}
        per_turn.append(pt)
    write_jsonl(OUT / "guard_ref_rows.jsonl", ref_rows)
    write_jsonl(OUT / "guard_turn_rows.jsonl", per_turn)
    rep: dict = {"n_turns": len(sel), "dialects": dict(collections.Counter(t["dialect"] for t in sel)),
                 "sample_errors": sum(1 for s in samples_all if "error" in s), "raw": raw,
                 "echo_errors": sum(1 for r in read_guard_echoes() if "error" in r)}
    raw["parse_rate"] = raw["n_parsed"] / max(1, raw["n_sampled"])
    raw["cap_rate"] = raw["n_cap"] / max(1, raw["n_sampled"])
    raw["leak_rate_parsed"] = raw["n_leak"] / max(1, raw["n_parsed"])
    raw["completion_tokens_per_ref"] = raw["completion_tokens"] / max(1, raw["n_sampled"])
    clean_rows = [r for r in ref_rows if not r["leak"]]
    rep["refs"] = {"g_in": aggregate_refs([r for r in clean_rows if r["in_band"]]),
                   "g_out": aggregate_refs([r for r in clean_rows if not r["in_band"]]),
                   "g_clean": aggregate_refs(clean_rows),
                   "g_all_parsed": aggregate_refs(ref_rows),
                   "g_leaked": aggregate_refs([r for r in ref_rows if r["leak"]])}
    rep["refs_by_dialect"] = {k: {"g_in": aggregate_refs([r for r in clean_rows if r["in_band"] and r["dialect"] == k]),
                                  "g_out": aggregate_refs([r for r in clean_rows if not r["in_band"] and r["dialect"] == k]),
                                  "g_clean": aggregate_refs([r for r in clean_rows if r["dialect"] == k])}
                              for k in ("bash", "tool_call", "terminus_json")}
    a = sum(1 for r in clean_rows if r["in_band"] and r["change"])
    b = sum(1 for r in clean_rows if r["in_band"] and not r["change"])
    c = sum(1 for r in clean_rows if not r["in_band"] and r["change"])
    d = sum(1 for r in clean_rows if not r["in_band"] and not r["change"])
    p1, p2 = a / max(1, a + b), c / max(1, c + d)
    pp = (a + c) / max(1, a + b + c + d)
    se = math.sqrt(pp * (1 - pp) * (1 / max(1, a + b) + 1 / max(1, c + d))) if 0 < pp < 1 else None
    rep["inband_x_change"] = {"in_change": a, "in_same": b, "out_change": c, "out_same": d,
                              "odds_ratio": ((a * d) / (b * c)) if b * c else None, "z_diff": ((p1 - p2) / se) if se else None}
    rep["headroom"] = {k: headroom_table(per_turn, k) for k in ("h_clean", "h_in", "h_in3", "h_out", "h_out3")}
    rep["headroom_by_dialect"] = {kd: {k: headroom_table([p for p in per_turn if p["dialect"] == kd], k) for k in ("h_clean", "h_in")}
                                  for kd in ("bash", "tool_call", "terminus_json")}
    cov = {f"n6_ge{m}": _rate([p["n_inband"] >= m for p in per_turn]) for m in (1, 2, 3)}
    cov.update({f"n6_ge{m}_anyleak": _rate([p["n_inband_any"] >= m for p in per_turn]) for m in (1, 2, 3)})
    cov["per_ref_inband_clean"] = _rate([r["in_band"] for r in clean_rows])
    cov["per_ref_inband_parsed"] = _rate([r["in_band"] for r in ref_rows])
    cov["inband_per_turn_mean"] = _mean([p["n_inband"] for p in per_turn])
    # paired vs the stored unguarded p_fr on the same turns (per-ref rates)
    cov["stored_pfr_inband_rate_same_turns"] = sum(p["pfr_inband"] for p in per_turn) / max(1, sum(p["pfr_parsed"] for p in per_turn))
    cov["stored_pfr_leak_rate_same_turns"] = sum(p["pfr_leak"] for p in per_turn) / max(1, sum(p["pfr_parsed"] for p in per_turn))
    cov["stored_pfr_k3_ge2_same_turns"] = _rate([p["pfr_inband"] >= 2 for p in per_turn])
    cov["stored_pfr_k3_ge3_same_turns"] = _rate([p["pfr_inband"] >= 3 for p in per_turn])
    cov["change_guard_minus_stored"] = _paired([p["g_change"] - p["pfr_change"] for p in per_turn
                                                if p["g_change"] is not None and p["pfr_change"] is not None])
    rep["coverage"] = cov
    jt = json.loads((OUT / "guard_judge_turns.json").read_text()) if (OUT / "guard_judge_turns.json").exists() else []
    rep["judge"] = {"n_turns": len(jt), "n": sum(1 for r in ref_rows if r["judge_same"] is not None),
                    "same_rate_clean": _rate([r["judge_same"] for r in clean_rows]),
                    "same_rate_in": _rate([r["judge_same"] for r in clean_rows if r["in_band"]]),
                    "same_rate_out": _rate([r["judge_same"] for r in clean_rows if not r["in_band"]])}
    return rep


# ------------------------------------------------------------------ report
def _f(x, w=6, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    if isinstance(x, dict):
        return _f(x.get("z"), w, 2)
    if isinstance(x, bool):
        return f"{str(x):>{w}}"
    return f"{x:{w}.{p}f}"


def ref_table(P, groups: dict[str, dict], title: str) -> None:
    P(f"-- {title}")
    P(f"{'group':<12} {'n':>5} {'change':>6} {'exactB':>6} {'jacB':>6} {'exactC':>6} {'jacC':>6} {'judge':>6} {'jn':>4} "
      f"{'leak':>6} {'leakZ':>6} {'lenZp50':>7} {'lenZmn':>6} {'lenYp50':>7} {'Bmean':>7} {'Bpass':>6} {'cap':>6} {'|z|p50':>6} {'zmean':>6} {'visib':>6}")
    for name, g in groups.items():
        P(f"{name:<12} {g['n']:>5} {_f(g['change'])} {_f(g['exact_blind'])} {_f(g['jac_blind'])} {_f(g['exact_ctrl'])} {_f(g['jac_ctrl'])} "
          f"{_f(g['judge_same'])} {g['judge_n']:>4} {_f(g['leak'])} {_f(g['leak_z'])} {_f(g['len_z_p50'],7,0)} {_f(g['len_z_mean'],6,0)} "
          f"{_f(g['len_y_p50'],7,0)} {_f(g['B_mean'],7,4)} {_f(g['B_pass'])} {_f(g['cap'])} {_f(g['abs_z_p50'],6,2)} {_f(g['z_mean'],6,2)} {_f(g['visible'])}")


def headroom_lines(P, H: dict, title: str) -> None:
    P(f"-- {title}  (R per byte; own needs |S|>=3; blind-own/king need |S|>=2; hz = paired z; d/σ = mean diff / σ_R(dialect))")
    P(f"{'set':<10} {'n>=2':>4} {'n>=3':>4} {'R_own':>8} {'R_bown3':>8} {'R_king3':>8} {'hz o-b':>6} {'d/σ':>5} {'hz o-k':>6} {'d/σ':>5} | {'R_bown':>8} {'R_king':>8} {'hz b-k':>6} {'d/σ':>5}")
    for k, h in H.items():
        P(f"{k:<10} {h['n_turns_ge2']:>4} {h['n_turns_ge3']:>4} {_f(h['R_own'],8,4)} {_f(h['R_bown_ge3'],8,4)} {_f(h['R_king_ge3'],8,4)} "
          f"{_f(h['hz_own_bown'])} {_f(h['eff_own_bown'],5,2)} {_f(h['hz_own_king'])} {_f(h['eff_own_king'],5,2)} | "
          f"{_f(h['R_bown'],8,4)} {_f(h['R_king'],8,4)} {_f(h['hz_bown_king'])} {_f(h['eff_bown_king'],5,2)}")


def cmd_stored(args: argparse.Namespace) -> None:
    rep = stored_analysis()
    (OUT / "stored_report.json").write_text(json.dumps(rep, indent=1, default=str))
    lines = []
    P = lines.append
    P("X1.1 — stored p_fr refs (k=3, unguarded) split by the live blind band")
    P(f"turns {rep['n_turns']}")
    P("Terms: change = norm-exact-different from all 3 blind refs AND best Jaccard < 0.5; exactB/jacB vs stored blind refs; exactC/jacC vs the fresh p_none refs;")
    P("       judge = glm-5.3-flash SAME rate on ref i vs blind ref i (i in 0,1; 60 turns); leak = >=30-char note substring in z or y; B per byte, pass >= 0.02;")
    P("       |z|p50 = median typicality z of the thought vs the blind band (live σ); visib = share of thoughts with a visible (post-</think>) part.")
    ref_table(P, rep["refs"], "refs pooled (in = in-band, out = out-of-band)")
    for k, g in rep["refs_by_dialect"].items():
        ref_table(P, g, f"refs dialect {k}")
    ic = rep["inband_x_change"]
    P(f"in-band x change 2x2: in&change {ic['in_change']} in&same {ic['in_same']} | out&change {ic['out_change']} out&same {ic['out_same']} "
      f"| odds ratio {_f(ic['odds_ratio'],6,2)} | z(change_in − change_out) {_f(ic['z_diff'],6,2)}")
    P(f"median |z| of changed refs {_f(rep['absz_by_change']['change'],6,2)} vs same-decision refs {_f(rep['absz_by_change']['same'],6,2)}")
    P("")
    headroom_lines(P, rep["headroom"], "headroom by reference set: all = 3 p_fr refs (original), in = in-band refs only, in_clean = in-band & no leak, out = out-of-band only")
    for kd, H in rep["headroom_by_dialect"].items():
        headroom_lines(P, H, f"headroom dialect {kd}")
    c = rep["coverage"]
    P("")
    P(f"coverage: per-ref in-band {c['per_ref_inband']:.3f} | k=3 turns with >=1 in-band {c['k3_ge1']:.3f}, >=2 {c['k3_ge2']:.3f}, >=3 {c['k3_ge3']:.3f} "
      f"(in-band & no-leak: {c['k3_ge1_clean']:.3f} / {c['k3_ge2_clean']:.3f} / {c['k3_ge3_clean']:.3f}) | p_none k=3 >=2 {c['p_none_k3_ge2']:.3f} >=3 {c['p_none_k3_ge3']:.3f}")
    P(f"projected N=6 (binomial): per-turn plug-in p_t: >=1 {c['n6_ge1_perturn']:.3f} >=2 {c['n6_ge2_perturn']:.3f} >=3 {c['n6_ge3_perturn']:.3f} | "
      f"shrunk p_t: {c['n6_ge1_shrunk']:.3f} / {c['n6_ge2_shrunk']:.3f} / {c['n6_ge3_shrunk']:.3f} | pooled p: {c['n6_ge1_pooled']:.3f} / {c['n6_ge2_pooled']:.3f} / {c['n6_ge3_pooled']:.3f}")
    txt = "\n".join(lines) + "\n"
    (OUT / "stored_report.txt").write_text(txt)
    print(txt)


def cmd_report(args: argparse.Namespace) -> None:
    cmd_stored(args)
    rep = guarded_analysis()
    rep["cost_usd_total"] = total_cost()
    (OUT / "guard_report.json").write_text(json.dumps(rep, indent=1, default=str))
    lines = []
    P = lines.append
    P("X1.2 — guarded resample: N=6 p_fr refs per turn with the paraphrase guard prefixed to the note")
    P(f"turns {rep['n_turns']} {rep['dialects']}; sample errors {rep['sample_errors']}, echo errors {rep['echo_errors']}; $ this probe (all stages) {rep['cost_usd_total']:.2f}")
    r = rep["raw"]
    P(f"sampled {r['n_sampled']} parsed {r['n_parsed']} ({r['parse_rate']:.3f}) cap {r['cap_rate']:.3f} think-closed {r['n_think_closed']} text-fallback {r['n_text_fb']} "
      f"compl tok/ref {r['completion_tokens_per_ref']:.0f} | leak among parsed {r['n_leak']} ({r['leak_rate_parsed']:.3f}; z {r['n_leak_z']}, y {r['n_leak_y']}; guard-text quoted {r['n_leak_guard']}) -> rejected")
    ref_table(P, rep["refs"], "guarded refs (clean = parsed & no leak; in/out split clean refs by the live band; judge on the 30-turn subsample, ref i vs blind i mod 3)")
    for k, g in rep["refs_by_dialect"].items():
        ref_table(P, g, f"guarded refs dialect {k}")
    ic = rep["inband_x_change"]
    P(f"in-band x change 2x2 (clean): in&change {ic['in_change']} in&same {ic['in_same']} | out&change {ic['out_change']} out&same {ic['out_same']} "
      f"| odds ratio {_f(ic['odds_ratio'],6,2)} | z(change_in − change_out) {_f(ic['z_diff'],6,2)}")
    j = rep["judge"]
    P(f"judge: {j['n']} comparisons on {j['n_turns']} turns; SAME rate clean {_f(j['same_rate_clean'])} in-band {_f(j['same_rate_in'])} out-of-band {_f(j['same_rate_out'])} "
      f"(p_none floor 0.675, stored p_fr 0.466)")
    P("")
    headroom_lines(P, rep["headroom"], "headroom: clean = all non-leak guarded refs (up to 6), in = in-band clean refs, in3 = first 3 in-band clean refs (k=3-matched), out/out3 = out-of-band")
    for kd, H in rep["headroom_by_dialect"].items():
        headroom_lines(P, H, f"headroom dialect {kd}")
    c = rep["coverage"]
    P("")
    P(f"coverage at N=6 (in-band & clean): >=1 {c['n6_ge1']:.3f} >=2 {c['n6_ge2']:.3f} >=3 {c['n6_ge3']:.3f} | in-band incl. leaked: {c['n6_ge1_anyleak']:.3f} / {c['n6_ge2_anyleak']:.3f} / {c['n6_ge3_anyleak']:.3f} "
      f"| in-band per clean ref {c['per_ref_inband_clean']:.3f} (per parsed ref {c['per_ref_inband_parsed']:.3f}); mean in-band clean refs/turn {c['inband_per_turn_mean']:.2f}")
    P(f"same 60 turns, stored unguarded p_fr (k=3): per-ref in-band {c['stored_pfr_inband_rate_same_turns']:.3f} leak {c['stored_pfr_leak_rate_same_turns']:.3f} turns >=2 in-band {c['stored_pfr_k3_ge2_same_turns']:.3f} >=3 {c['stored_pfr_k3_ge3_same_turns']:.3f}")
    P(f"change rate guarded − stored p_fr (paired over turns): mean {_f(c['change_guard_minus_stored'].get('mean'),6,3)} z {_f(c['change_guard_minus_stored'])}")
    txt = "\n".join(lines) + "\n"
    (OUT / "guard_report.txt").write_text(txt)
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stored")
    a.set_defaults(fn=cmd_stored)
    b = sub.add_parser("sample")
    b.add_argument("--turns", type=int, default=60)
    b.add_argument("--concurrency", type=int, default=24)
    b.set_defaults(fn=cmd_sample)
    c = sub.add_parser("echo")
    c.add_argument("--concurrency", type=int, default=24)
    c.add_argument("--shard", type=int, default=0)
    c.add_argument("--nshards", type=int, default=1)
    c.set_defaults(fn=cmd_echo)
    d = sub.add_parser("judge")
    d.add_argument("--turns", type=int, default=30)
    d.set_defaults(fn=cmd_judge)
    e = sub.add_parser("report")
    e.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
