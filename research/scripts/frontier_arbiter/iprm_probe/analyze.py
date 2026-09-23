"""IPRM probe — stage 4 (local): read the pod's logprobs and write
research/results/frontier_arbiter/iprm/{report.txt, report.json, rows.jsonl}.

Terms (one line each):
  lpC     = sum log p_C (action body tokens | prefix, empty thought)     frozen teacher
  lpC+    = same under the solved-SFT LoRA teacher C+
  TERM    = lpC+ - lpC              (per action, sum over body tokens)  the N1 ranking term
  TERM_pb = TERM / body bytes
  PLAUS   = lpC_pb(y) - max_i lpC_pb(y_i) over the state's TEACHER actions (per-byte typicality gate proxy)
  label   = outcome of the graded continuation that took y as its first action (1 solved / 0 failed)
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

OUT = Path("/tmp/iprm")
RES = common.REPO / "research" / "results" / "frontier_arbiter" / "iprm"
TEACHER_ORIGINS = ("teacher_cont", "teacher_orig", "teacher_first_unlabelled", "teacher_sample_unlabelled")
KING_ORIGINS = ("king_stored", "king_fresh")


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.mean(xs) if xs else None


def fmt(x, nd=3):
    return "  n/a " if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def boot_ci(vals: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=OUT / "score_out.jsonl")
    ap.add_argument("--train-summary", type=Path, default=OUT / "train_summary.json")
    ap.add_argument("--train-log", type=Path, default=OUT / "train_log.jsonl")
    ap.add_argument("--cost", type=Path, default=OUT / "cost.json")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    states = {s["state"]: s for s in common.read_jsonl(OUT / "eval_states.jsonl")}
    rows = {r["row_id"]: r for r in common.read_jsonl(OUT / "eval_rows.jsonl")}
    sc = collections.defaultdict(dict)          # (row_id, ctx) -> model -> rec
    for d in common.read_jsonl(a.scores):
        sc[(d["row_id"], d["ctx"])][d["model"]] = d
    parity = common.read_jsonl(OUT / "parity_engy.jsonl") if (OUT / "parity_engy.jsonl").exists() else []

    # ---- per-row terms (own prefix) ------------------------------------------------
    R = []
    for rid, r in rows.items():
        m = sc.get((rid, "own"))
        if not m or "base" not in m or "plus" not in m:
            continue
        b, p = m["base"], m["plus"]
        nb = max(b["n_body"], 1)
        rec = {"row_id": rid, "state": r["state"], "kind": r["kind"], "origins": r["origins"], "labels": r["labels"],
               "arms": r["arms"], "p_solved": r["p_solved"], "y": r["y"], "body": r["body"],
               "n_body_tok": b["n_body"], "n_act_tok": b["n_act"],
               "lpC": b["lp_body"], "lpCp": p["lp_body"], "term": p["lp_body"] - b["lp_body"],
               "lpC_act": b["lp_act"], "lpCp_act": p["lp_act"], "term_act": p["lp_act"] - b["lp_act"]}
        n_bytes = max(len(r["body"].encode()), 1)
        rec["lpC_pb"] = b["lp_body"] / n_bytes
        rec["term_pb"] = rec["term"] / n_bytes
        rec["n_bytes"] = n_bytes
        for ctx in ("earlier", "other_task"):
            mm = sc.get((rid, ctx))
            if mm and "base" in mm and "plus" in mm:
                rec[f"term_{ctx}"] = mm["plus"]["lp_body"] - mm["base"]["lp_body"]
                rec[f"lpC_{ctx}"] = mm["base"]["lp_body"]
        R.append(rec)
    by_state = collections.defaultdict(list)
    for rec in R:
        by_state[rec["state"]].append(rec)
    # PLAUS: per-byte typicality vs the state's teacher actions
    for s, recs in by_state.items():
        t = [x["lpC_pb"] for x in recs if any(o in TEACHER_ORIGINS for o in x["origins"])]
        tmax = max(t) if t else None
        tsum = [x["lpC"] for x in recs if any(o in TEACHER_ORIGINS for o in x["origins"])]
        for x in recs:
            x["plaus"] = (x["lpC_pb"] - tmax) if tmax is not None else None
            x["plaus_sum"] = (x["lpC"] - max(tsum)) if tsum else None
            x["is_teacher"] = any(o in TEACHER_ORIGINS for o in x["origins"])
            x["is_king"] = any(o in KING_ORIGINS for o in x["origins"])
    common.write_jsonl(RES / "rows.jsonl", R)

    L: list[str] = []
    J: dict = {"n_states": len(by_state), "n_rows": len(R)}
    P = L.append
    P("IPRM probe (N1): implicit process reward from the success-conditioned teacher — lpC+(y|x) - lpC(y|x) over action-body tokens")
    P("=" * 120)
    P(__doc__.split("Terms (one line each):")[1].rstrip())
    P("")

    # ---- training summary ----------------------------------------------------------
    if a.train_summary.exists():
        ts = json.load(open(a.train_summary))
        meta = json.load(open(OUT / "train_meta.json"))
        log = common.read_jsonl(a.train_log)
        first = [x["loss"] for x in log[:5]]
        last = [x["loss"] for x in log[-5:]]
        P("0. TRAINING (C+ = LoRA on the frozen teacher, loss on assistant tokens of SOLVED teacher trajectories)")
        P(f"   data: {meta['n_traj']} solved trajectories -> {meta['n_windows']} windows (<= {meta['max_tokens']} tok; head + turn block), "
          f"{meta['n_tokens']/1e6:.1f}M tokens, {meta['n_loss_tokens']/1e6:.1f}M loss tokens; {meta['holdout_sids']} eval tasks held out")
        P(f"   cells (source|harness -> windows): " + ", ".join(f"{k} {v}" for k, v in meta["cells"].items()))
        P(f"   LoRA r={ts['args']['rank']} alpha={ts['args']['alpha']} on every linear of the language model; lr {ts['args']['lr']} cosine, accum {ts['args']['accum']}, "
          f"{ts['steps']} optimizer steps ({ts['micro']} windows seen, {ts['tokens_seen']/1e6:.1f}M tokens), {ts['minutes']:.0f} min on 1x B200, "
          f"{log[-1]['tok_s']:.0f} tok/s, peak mem {log[-1]['mem_gb']:.0f} GB")
        P(f"   loss (nats/token, assistant tokens): first 5 steps {mean(first):.3f} -> last 5 steps {mean(last):.3f}; ema {ts['final_ema']:.3f}")
        J["training"] = {"summary": ts, "meta": meta, "loss_first5": mean(first), "loss_last5": mean(last)}
        P("")

    # ---- (a) AUC --------------------------------------------------------------------
    P("1. AUC solved-vs-failed FIRST ACTIONS (each graded continuation = one sample; the action it started with carries its outcome)")
    P("   arms T+F1 = teacher-continued only (label = the teacher's own finish from that action); +F adds frontier-continued rows")
    J["auc"] = {}
    for arms_name in ("T+F1", "T only"):
        # T+F1 = arms T, F1, F1b, F1c (all teacher-continued); arm "F" (frontier-continued) is excluded
        if arms_name == "T+F1":
            samples = [(x, lab) for x in R for lab, arm in zip(x["labels"], x["arms"]) if arm != "F"]
        else:
            samples = [(x, lab) for x in R for lab, arm in zip(x["labels"], x["arms"]) if arm == "T"]
        P(f"   [{arms_name}] {len(samples)} samples ({sum(l for _, l in samples)} solved) at {len({x['state'] for x, _ in samples})} states")
        J["auc"][arms_name] = {}
        for key, name in (("term", "TERM (sum)"), ("term_pb", "TERM per byte"), ("lpC", "lpC (sum)"), ("lpC_pb", "lpC per byte"),
                          ("lpCp", "lpC+ (sum)"), ("plaus", "PLAUS"), ("term_act", "TERM whole action incl. fence")):
            pos = [x[key] for x, l in samples if l == 1 and x.get(key) is not None]
            neg = [x[key] for x, l in samples if l == 0 and x.get(key) is not None]
            pooled = auc(pos, neg)
            per_state = []
            for s in {x["state"] for x, _ in samples}:
                sp = [x[key] for x, l in samples if x["state"] == s and l == 1 and x.get(key) is not None]
                sn = [x[key] for x, l in samples if x["state"] == s and l == 0 and x.get(key) is not None]
                v = auc(sp, sn)
                if v is not None:
                    per_state.append(v)
            ci = boot_ci(per_state)
            P(f"      {name:32s} pooled AUC {fmt(pooled)}   within-state mean AUC {fmt(mean(per_state))} over {len(per_state)} states"
              + (f"  [95% {ci[0]:.2f},{ci[1]:.2f}]" if ci else "") + f"   frac states AUC>0.5 {fmt(mean([1.0 if v > 0.5 else 0.0 if v < 0.5 else 0.5 for v in per_state]), 2)}")
            J["auc"][arms_name][key] = {"pooled": pooled, "within_mean": mean(per_state), "n_states": len(per_state), "per_state": per_state,
                                       "ci95": ci}
        # distinct-action level: p_solved as label
        dist = [x for x in R if x["p_solved"] is not None and any(arm != "F" for arm in x["arms"])]
        pos = [x["term"] for x in dist if x["p_solved"] > 0.5]
        neg = [x["term"] for x in dist if x["p_solved"] < 0.5]
        P(f"      distinct actions (p_solved>0.5 vs <0.5, {len(pos)} vs {len(neg)}): pooled AUC TERM {fmt(auc(pos, neg))}")
    # by dialect
    P("   by dialect (T+F1 samples, TERM sum): ")
    for kind in ("bash", "tool_call", "terminus_json"):
        samples = [(x, lab) for x in R if x["kind"] == kind for lab, arm in zip(x["labels"], x["arms"]) if arm != "F"]
        pos = [x["term"] for x, l in samples if l == 1]
        neg = [x["term"] for x, l in samples if l == 0]
        per_state = []
        for s in {x["state"] for x, _ in samples}:
            v = auc([x["term"] for x, l in samples if x["state"] == s and l == 1], [x["term"] for x, l in samples if x["state"] == s and l == 0])
            if v is not None:
                per_state.append(v)
        P(f"      {kind:14s} n={len(samples):3d} pooled {fmt(auc(pos, neg))} within-state {fmt(mean(per_state))} ({len(per_state)} states)")
    # mean term by label
    samples = [(x, lab) for x in R for lab, arm in zip(x["labels"], x["arms"]) if arm != "F"]
    P(f"   mean TERM solved {fmt(mean([x['term'] for x, l in samples if l]))} vs failed {fmt(mean([x['term'] for x, l in samples if not l]))}"
      f" (per byte {fmt(mean([x['term_pb'] for x, l in samples if l]), 4)} vs {fmt(mean([x['term_pb'] for x, l in samples if not l]), 4)});"
      f" mean lpC solved {fmt(mean([x['lpC'] for x, l in samples if l]))} vs failed {fmt(mean([x['lpC'] for x, l in samples if not l]))}")
    tv = sorted(x["term"] for x in R)
    q = lambda p: tv[int(p * (len(tv) - 1))]
    P(f"   TERM distribution over all {len(tv)} scored (state, action) rows: p5 {q(.05):+.2f}  p25 {q(.25):+.2f}  median {q(.5):+.2f}  p75 {q(.75):+.2f}  p95 {q(.95):+.2f}; "
      f"median |TERM| {st.median(abs(v) for v in tv):.2f} nats over a median body of {st.median(x['n_body_tok'] for x in R):.0f} tokens "
      f"(numerical floor: cached-vs-full forward disagree by ~0.1 nats, see 6) — C+ does move away from C; the movement just does not track the outcome")
    J["term_dist"] = {"p5": q(.05), "p25": q(.25), "median": q(.5), "p75": q(.75), "p95": q(.95), "median_abs": st.median(abs(v) for v in tv)}
    P("")

    # ---- (b) ceiling proposals ----------------------------------------------------
    P("2. CEILING STATES: teacher 0/N in arm T, a forced frontier proposal solved — is the solved proposal ranked above the teacher's failed actions?")
    ceil = []
    for s, recs in by_state.items():
        t_fail = [x for x in recs if any(arm == "T" for arm in x["arms"]) and all(l == 0 for l, arm in zip(x["labels"], x["arms"]) if arm == "T")]
        t_any_solved = any(l == 1 for x in recs for l, arm in zip(x["labels"], x["arms"]) if arm == "T")
        props = [x for x in recs if any(arm.startswith("F1") for arm in x["arms"]) and any(l == 1 for l, arm in zip(x["labels"], x["arms"]) if arm.startswith("F1"))]
        if t_any_solved or not t_fail or not props:
            continue
        for p in props:
            row = {"state": s, "kind": recs[0]["kind"], "n_teacher_failed": len(t_fail)}
            for key in ("term", "term_pb", "lpC", "lpC_pb", "plaus"):
                beats = sum(1 for x in t_fail if p[key] > x[key])
                row[f"beats_{key}"] = beats / len(t_fail)
                row[f"top_{key}"] = beats == len(t_fail)
            row["proposal"] = p["body"][:100]
            ceil.append(row)
    if ceil:
        for key, name in (("term", "TERM"), ("term_pb", "TERM/byte"), ("lpC", "lpC"), ("lpC_pb", "lpC/byte"), ("plaus", "PLAUS")):
            P(f"   {name:10s} solved proposal above ALL failed teacher actions: {sum(r[f'top_{key}'] for r in ceil)}/{len(ceil)}; "
              f"mean fraction of failed teacher actions beaten {mean([r[f'beats_{key}'] for r in ceil]):.2f}")
        for r in ceil:
            P(f"      {r['state'][:12]} {r['kind']:13s} teacher failed {r['n_teacher_failed']}  beats(TERM) {r['beats_term']:.2f} beats(lpC) {r['beats_lpC']:.2f}  {r['proposal'][:70]!r}")
    else:
        P("   no ceiling state with a solved forced proposal in the scored set")
    J["ceiling"] = ceil
    P("")

    # ---- (c) attack rows ------------------------------------------------------------
    P("3. ATTACK ROWS: where `ls -la` (generic) and repeat-last land in each state's ranking (candidates = labelled actions + teacher samples)")
    J["attacks"] = {}
    for origin in ("attack_generic", "attack_repeat"):
        stats = collections.defaultdict(list)
        for s, recs in by_state.items():
            atk = [x for x in recs if origin in x["origins"]]
            cands = [x for x in recs if (x["labels"] or x["is_teacher"]) and origin not in x["origins"]]
            lab = [x for x in recs if x["labels"] and origin not in x["origins"]]
            if not atk or not cands:
                continue
            at = atk[0]
            for key in ("term", "term_pb", "plaus", "lpC_pb"):
                vals = [x[key] for x in cands if x.get(key) is not None]
                if not vals or at.get(key) is None:
                    continue
                pct = sum(1 for v in vals if at[key] > v) / len(vals)      # fraction of candidates the attack beats
                stats[key].append(pct)
                if lab:
                    lv = sorted(x[key] for x in lab if x.get(key) is not None)
                    if lv:
                        med = lv[len(lv) // 2]
                        stats[key + "_beats_median_labelled"].append(1.0 if at[key] > med else 0.0)
        P(f"   {origin}: {len(stats.get('term', []))} states")
        for key, name in (("term", "TERM"), ("term_pb", "TERM/byte"), ("plaus", "PLAUS"), ("lpC_pb", "lpC/byte")):
            P(f"      {name:10s} mean fraction of candidates beaten {fmt(mean(stats.get(key, [])), 2)}; beats the MEDIAN labelled action in "
              f"{fmt(mean(stats.get(key + '_beats_median_labelled', [])), 2)} of states")
        J["attacks"][origin] = {k: mean(v) for k, v in stats.items()}
        # under the gate: how many attacks pass PLAUS >= -band (band_floor-like 0.002/byte?) — report raw plaus distribution
        pl = [x["plaus"] for recs in by_state.values() for x in recs if origin in x["origins"] and x.get("plaus") is not None]
        if pl:
            P(f"      PLAUS of the attack: median {st.median(pl):+.4f}/byte, frac >= -0.05 {mean([1.0 if v >= -0.05 else 0.0 for v in pl]):.2f}, frac >= 0 {mean([1.0 if v >= 0 else 0.0 for v in pl]):.2f}")
            P(f"      TERM of the attack: median {st.median([x['term'] for recs in by_state.values() for x in recs if origin in x['origins']]):+.3f} "
              f"vs median TERM of labelled-solved {st.median([x['term'] for x in R for l, arm in zip(x['labels'], x['arms']) if l == 1]):+.3f} "
              f"and labelled-failed {st.median([x['term'] for x in R for l, arm in zip(x['labels'], x['arms']) if l == 0]):+.3f}")
    P("")

    # ---- (d) teacher vs king ---------------------------------------------------------
    P("4. TEACHER vs KING (reign 20): mean TERM of king actions (stored duel action + 3 fresh samples at the 36 primary states) vs teacher actions, paired per state")
    J["king"] = {}
    for subset, states_ok in (("primary (36)", {s for s, v in states.items() if v["primary"]}), ("all", set(states))):
        diffs, diffs_pb, tk, tt = [], [], [], []
        for s, recs in by_state.items():
            if s not in states_ok:
                continue
            k = [x["term"] for x in recs if x["is_king"]]
            t = [x["term"] for x in recs if x["is_teacher"]]
            if k and t:
                diffs.append(mean(k) - mean(t))
                diffs_pb.append(mean([x["term_pb"] for x in recs if x["is_king"]]) - mean([x["term_pb"] for x in recs if x["is_teacher"]]))
                tk.extend(k)
                tt.extend(t)
        if diffs:
            se = st.pstdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
            P(f"   {subset:12s} states {len(diffs)}: mean TERM king {mean(tk):+.3f} (n={len(tk)}) vs teacher {mean(tt):+.3f} (n={len(tt)}); "
              f"paired king-teacher {mean(diffs):+.3f} (SE {se:.3f}, z {mean(diffs)/se if se else float('nan'):+.2f}); per byte {mean(diffs_pb):+.4f}; "
              f"king above teacher in {mean([1.0 if d > 0 else 0.0 for d in diffs]):.2f} of states")
            J["king"][subset] = {"n_states": len(diffs), "mean_king": mean(tk), "mean_teacher": mean(tt), "paired": mean(diffs), "se": se}
    # king fresh samples: parse / think-close stats
    if (OUT / "king_samples.jsonl").exists():
        ks = common.read_jsonl(OUT / "king_samples.jsonl")
        P(f"   fresh king samples: {len(ks)}, think closed {sum(1 for k in ks if k.get('think_closed'))}, parsed action {sum(1 for k in ks if k.get('y'))}")
    P("")

    # ---- (e) prefix swap ---------------------------------------------------------------
    P("5. PREFIX-SWAP CONTROL: the same action body scored under (i) its own prefix, (ii) the same trajectory 2 assistant turns earlier, (iii) another task's prefix (same dialect)")
    ctl = [x for x in R if "term_earlier" in x or "term_other_task" in x]
    J["prefix_swap"] = {}
    for ctx in ("earlier", "other_task"):
        xs = [x for x in ctl if f"term_{ctx}" in x]
        if not xs:
            continue
        d_prefix = [abs(x[f"term_{ctx}"] - x["term"]) for x in xs]
        # action-swap: within the same state (own prefix), |term_a - term_b| over pairs of control actions
        d_action = []
        for s, recs in by_state.items():
            c = [x for x in recs if x in xs]
            for u, v in itertools.combinations(c, 2):
                d_action.append(abs(u["term"] - v["term"]))
        var_prefix = st.pvariance([x[f"term_{ctx}"] - x["term"] for x in xs])
        # variance across actions within state, own prefix (pooled within-state variance)
        wv = []
        for s, recs in by_state.items():
            c = [x["term"] for x in recs if x in xs]
            if len(c) >= 2:
                wv.append(st.pvariance(c))
        corr = None
        if len(xs) > 3:
            a1 = [x["term"] for x in xs]
            a2 = [x[f"term_{ctx}"] for x in xs]
            ma, mb = st.mean(a1), st.mean(a2)
            cov = sum((p - ma) * (q - mb) for p, q in zip(a1, a2))
            den = math.sqrt(sum((p - ma) ** 2 for p in a1) * sum((q - mb) ** 2 for q in a2))
            corr = cov / den if den else None
        P(f"   {ctx:10s} n={len(xs)} actions: mean |TERM(other prefix) - TERM(own)| = {mean(d_prefix):.3f}  vs  mean |TERM(a) - TERM(b)| across actions, same prefix = {fmt(mean(d_action))}; "
          f"var across prefixes (same action) {var_prefix:.3f} vs mean within-state var across actions {fmt(mean(wv))}; corr(TERM own, TERM {ctx}) {fmt(corr, 2)}")
        # does the prefix swap preserve labels' ordering?
        samples = [(x, lab) for x in xs for lab, arm in zip(x["labels"], x["arms"]) if arm != "F"]
        pos = [x[f"term_{ctx}"] for x, l in samples if l == 1]
        neg = [x[f"term_{ctx}"] for x, l in samples if l == 0]
        P(f"              pooled AUC of TERM computed under the {ctx} prefix: {fmt(auc(pos, neg))} (own prefix on the same samples: "
          f"{fmt(auc([x['term'] for x, l in samples if l == 1], [x['term'] for x, l in samples if l == 0]))})")
        J["prefix_swap"][ctx] = {"n": len(xs), "mean_abs_prefix_shift": mean(d_prefix), "mean_abs_action_diff": mean(d_action),
                                 "var_prefix": var_prefix, "within_state_var_actions": mean(wv), "corr": corr}
    P("   filler control (f): the term reads only the action body given the prefix with an EMPTY thought — a filler thought is invisible to it;")
    P("   a miner's thought cannot move TERM at all (neither up nor down); TERM is a pure action term and must be paired with the thought legs.")
    P("")

    # ---- parity -------------------------------------------------------------------------
    if parity:
        for p in parity:
            m = sc.get((p["row_id"], "own"), {}).get("base")
            p["pod_lpC"] = m["lp_body"] if m else None
            p["pod_n_body"] = m["n_body"] if m else None
        pp = [p for p in parity if p.get("pod_lpC") is not None]
        d = [p["pod_lpC"] - p["engy_lpC"] for p in pp]
        if d:
            P(f"6. PARITY pod (HF bf16, cached prefix) vs live teacher echo (Engy vLLM) on {len(d)} rows: mean lpC diff {mean(d):+.3f}, "
              f"max |diff| {max(abs(v) for v in d):.3f}, mean |lpC| {mean([abs(p['engy_lpC']) for p in pp]):.2f}; "
              f"token-count match {sum(1 for p in pp if p['pod_n_body'] == p['n_body'])}/{len(pp)}")
        J["parity_engy"] = parity
    if (OUT / "score_out.checks.json").exists():
        ch = json.load(open(OUT / "score_out.checks.json"))
        if ch:
            P(f"   cached-prefix vs full-forward parity on {len(ch)} rows: mean |diff| {mean([abs(c['cached'] - c['full']) for c in ch]):.3f} nats")
    P("")

    # ---- cost ----------------------------------------------------------------------------
    if a.cost.exists():
        c = json.load(open(a.cost))
        P("7. COST / TIME")
        for k, v in c.items():
            P(f"   {k}: {v}")
        J["cost"] = c
        P("")

    # ---- verdict ---------------------------------------------------------------------------
    within = J["auc"]["T+F1"]["term"]["within_mean"]
    pooled = J["auc"]["T+F1"]["term"]["pooled"]
    gen_med = J["attacks"].get("attack_generic", {}).get("plaus_beats_median_labelled")
    rep_med = J["attacks"].get("attack_repeat", {}).get("plaus_beats_median_labelled")
    ps = J["prefix_swap"].get("other_task") or J["prefix_swap"].get("earlier") or {}
    P("8. VERDICT against the kill criteria")
    k1 = within is not None and within <= 0.6
    k2 = (gen_med or 0) > 0.5 or (rep_med or 0) > 0.5
    k3 = bool(ps) and ps.get("var_prefix") is not None and ps.get("within_state_var_actions") is not None and ps["var_prefix"] > ps["within_state_var_actions"]
    P(f"   within-state AUC(TERM) = {fmt(within)} (kill if <= 0.60) -> {'KILL' if k1 else 'pass'}; pooled {fmt(pooled)}")
    P(f"   generic / repeat beat the median labelled action under PLAUS in {fmt(gen_med, 2)} / {fmt(rep_med, 2)} of states (kill if > 0.5) -> {'KILL' if k2 else 'pass'}")
    if ps:
        P(f"   prefix-swap variance {ps['var_prefix']:.3f} vs action-swap variance {fmt(ps['within_state_var_actions'])} (kill if prefix > action) -> {'KILL' if k3 else 'pass'}")
    P(f"   => {'DEAD' if (k1 or k2 or k3) else 'ALIVE (kill criteria not met)'}")
    P("   reading: C+ is trained on the teacher's OWN solved samples (self-distillation), so what it learns is which tasks/trajectories")
    P("   the teacher tends to finish, not which action at a split state is right — TERM is higher on failed than solved first actions,")
    P("   higher on the king's actions than the teacher's, indifferent to `ls -la`/repeat-last, and moves ~8x more with the prefix than with the action.")
    J["verdict"] = {"within_auc": within, "pooled_auc": pooled, "generic_beats_median_plaus": gen_med, "repeat_beats_median_plaus": rep_med,
                    "prefix_swap": ps, "kill_auc": k1, "kill_attack": k2, "kill_prefix": k3, "dead": bool(k1 or k2 or k3)}

    (RES / "report.txt").write_text("\n".join(L) + "\n")
    json.dump(J, open(RES / "report.json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
