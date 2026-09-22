"""Fixed-byte A on the kings' real bench steps (2026-09-10).

The isomorphism question for the staged A leg: does A go up when the agent
actually does better? Re-reads the two on-policy bench probes
(v6_bench_onpolicy.json: 1,348 steps / 150 mini-swe runs of reigns 0–5;
v6_bench_obsdep.json: 1,600 steps with repeat / dead-tail labels) and
recomputes the A leg with the staged normalization

    b_i = (per-byte lift) · len_y / L0,   A = LME_tau(b_i),   L0 = 128

next to the per-byte A those probes reported. onpolicy rows carry the
per-ref `b`, so the recomputation is exact; obsdep rows carry only the
aggregated A, so there A_fixed = A · len_y / L0 (exact up to a per-step tau
rescaling — LME_tau(c·b) = c·LME_{tau/c}(b) — flagged as approximate).

Tests, in order of how much they matter for a flip decision:
  1. within-task run outcome: tasks that some reigns solved and others
     failed; AUC = P(solving run's mean leg > failing run's) over all
     solved/failed run pairs of the same task. 0.5 = blind.
  2. step pathologies: repeat steps (exact command repeat inside a run) and
     dead-tail steps (last 3 of a run that hit a limit) vs healthy steps.
  3. per-reign drift across the crowned lineage, and short/long bias.

    python research/scripts/v6_bench_renorm.py > research/results/v6_bench_renorm.txt
"""
from __future__ import annotations

import itertools
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
L0 = 128.0


def lme(vals: list[float], tau: float | None) -> float:
    if len(vals) == 1 or tau is None or tau <= 0:
        return st.mean(vals)
    m = max(vals)
    return m + tau * math.log(st.mean(math.exp((v - m) / tau) for v in vals))


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def zstat(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return (st.mean(a) - st.mean(b)) / se if se > 0 else float("nan")


def within_task_auc(rows: list[dict], leg: str) -> tuple[float, int, int]:
    by_run: dict[tuple, list[float]] = defaultdict(list)
    solved: dict[tuple, bool] = {}
    for r in rows:
        k = (r["label"], r["task"])
        by_run[k].append(r[leg])
        solved[k] = bool(r["resolved"])
    run_mean = {k: st.mean(v) for k, v in by_run.items()}
    tasks = {k[1] for k in run_mean}
    pos_all, pairs, n_mixed = 0.0, 0, 0
    for t in sorted(tasks):
        runs = [k for k in run_mean if k[1] == t]
        s = [run_mean[k] for k in runs if solved[k]]
        f = [run_mean[k] for k in runs if not solved[k]]
        if not s or not f:
            continue
        n_mixed += 1
        for a, b in itertools.product(s, f):
            pos_all += 1.0 if a > b else 0.5 if a == b else 0.0
            pairs += 1
    return (pos_all / pairs if pairs else float("nan")), pairs, n_mixed


def add_fixed(rows: list[dict], tau: float, exact: bool) -> None:
    for r in rows:
        if exact:
            r["A_fixed"] = lme([b * r["len_y"] / L0 for b in r["b"]], tau)
        else:
            r["A_fixed"] = r["A"] * r["len_y"] / L0
        r["minRG"] = min(r["R"], r["G"])
        r["minRGA_pb"] = min(r["R"], r["G"], r["A"])
        r["minRGA_fx"] = min(r["R"], r["G"], r["A_fixed"])


def main() -> int:
    out: list[str] = []
    p = out.append
    LEGS = ("R", "G", "A", "A_fixed", "minRG", "minRGA_pb", "minRGA_fx")

    # ---------------- onpolicy (exact) ----------------
    d = json.load(open(REPO / "research/results/v6_bench_onpolicy.json"))
    rows = d["rows"]
    add_fixed(rows, d["tau"], exact=True)
    p(f"== on-policy bench steps (exact recomputation): {len(rows)} steps, {d['n_runs']} runs, L0={L0:g} ==")
    p("")
    p("per reign (mean over steps)")
    p(f"{'reign':9} {'n':>5} " + " ".join(f"{l:>10}" for l in LEGS) + f" {'A<0':>6} {'Afx<0':>6}")
    for lab in sorted({r["label"] for r in rows}):
        g = [r for r in rows if r["label"] == lab]
        p(f"{lab:9} {len(g):5d} " + " ".join(f"{st.mean(r[l] for r in g):10.4f}" for l in LEGS)
          + f" {sum(r['A'] < 0 for r in g) / len(g):6.1%} {sum(r['A_fixed'] < 0 for r in g) / len(g):6.1%}")
    p("")
    p("within-task run outcome (mixed-outcome tasks): AUC = P(solving run > failing run)")
    p(f"{'leg':10} {'AUC':>6} {'pairs':>6} {'tasks':>6}")
    for l in LEGS:
        a, n, t = within_task_auc(rows, l)
        p(f"{l:10} {a:6.3f} {n:6d} {t:6d}")
    p("")
    p("pooled resolved vs unresolved runs (run-level means; confounded by task difficulty)")
    by_run: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_run[(r["label"], r["task"])].append(r)
    p(f"{'leg':10} {'resolved':>9} {'unresolved':>11} {'diff':>8} {'z':>6} {'AUC':>6}")
    for l in LEGS:
        res = [st.mean(x[l] for x in g) for g in by_run.values() if g[0]["resolved"]]
        unr = [st.mean(x[l] for x in g) for g in by_run.values() if not g[0]["resolved"]]
        p(f"{l:10} {st.mean(res):9.4f} {st.mean(unr):11.4f} {st.mean(res) - st.mean(unr):+8.4f} "
          f"{zstat(res, unr):6.2f} {auc(res, unr):6.3f}")
    p("")
    p("step-level by run exit status")
    p(f"{'exit_status':28} {'n':>5} {'A':>8} {'A_fixed':>8} {'minRGA_fx':>10}")
    for es in sorted({r["exit_status"] for r in rows}, key=lambda e: -sum(1 for r in rows if r["exit_status"] == e)):
        g = [r for r in rows if r["exit_status"] == es]
        p(f"{es:28} {len(g):5d} {st.mean(r['A'] for r in g):8.4f} {st.mean(r['A_fixed'] for r in g):8.4f} "
          f"{st.mean(r['minRGA_fx'] for r in g):10.4f}")
    p("")
    med_len = st.median(r["len_y"] for r in rows)
    sh = [r for r in rows if r["len_y"] <= med_len]
    lo = [r for r in rows if r["len_y"] > med_len]
    p(f"short/long bias (median len {med_len:.0f} chars): A per-byte {st.median(r['A'] for r in sh):.4f} vs "
      f"{st.median(r['A'] for r in lo):.4f} (ratio {st.median(r['A'] for r in sh) / st.median(r['A'] for r in lo):.2f}); "
      f"A_fixed {st.median(r['A_fixed'] for r in sh):.4f} vs {st.median(r['A_fixed'] for r in lo):.4f} "
      f"(ratio {st.median(r['A_fixed'] for r in sh) / st.median(r['A_fixed'] for r in lo):.2f})")
    p("")

    # ---------------- obsdep (approximate) ----------------
    d2 = json.load(open(REPO / "research/results/v6_bench_obsdep.json"))
    rows2 = d2["rows"]
    add_fixed(rows2, d2["tau"], exact=False)
    LEGS2 = ("R", "G", "A", "A_fixed", "B", "O_y", "minRG", "minRGA_pb", "minRGA_fx")
    p(f"== obsdep bench steps (A_fixed = A·len/L0, approximate): {len(rows2)} steps, "
      f"{sum(1 for r in rows2 if r['repeat'])} repeat, {sum(1 for r in rows2 if r['dead_tail'])} dead-tail ==")
    p("")
    p("step pathologies: z = (healthy − failing)/SE, AUC = P(healthy step > failing step)")
    p(f"{'leg':10} {'repeat':>9} {'non-rep':>9} {'z':>6} {'AUC':>6} | {'dead tail':>9} {'submitted':>9} {'z':>6} {'AUC':>6}")
    for l in LEGS2:
        have = [r for r in rows2 if r.get(l) is not None]
        rep = [r[l] for r in have if r["repeat"]]
        non = [r[l] for r in have if not r["repeat"]]
        dead = [r[l] for r in have if r["dead_tail"]]
        sub = [r[l] for r in have if r["submitted_run"] and not r["dead_tail"]]
        p(f"{l:10} {st.mean(rep):9.4f} {st.mean(non):9.4f} {zstat(non, rep):6.2f} {auc(non, rep):6.3f} | "
          f"{st.mean(dead):9.4f} {st.mean(sub):9.4f} {zstat(sub, dead):6.2f} {auc(sub, dead):6.3f}")
    p("")
    p("within-task run outcome (mixed tasks)")
    p(f"{'leg':10} {'AUC':>6} {'pairs':>6} {'tasks':>6}")
    for l in LEGS2:
        a, n, t = within_task_auc([r for r in rows2 if r.get(l) is not None], l)
        p(f"{l:10} {a:6.3f} {n:6d} {t:6d}")
    p("")
    p("per reign (mean over steps)")
    p(f"{'reign':9} {'n':>5} " + " ".join(f"{l:>10}" for l in ("A", "A_fixed", "minRG", "minRGA_fx")) + f" {'repeat%':>8}")
    for lab in sorted({r["label"] for r in rows2}):
        g = [r for r in rows2 if r["label"] == lab]
        p(f"{lab:9} {len(g):5d} " + " ".join(f"{st.mean(r[l] for r in g):10.4f}" for l in ("A", "A_fixed", "minRG", "minRGA_fx"))
          + f" {sum(1 for r in g if r['repeat']) / len(g):8.1%}")
    p("")
    med_len = st.median(r["len_y"] for r in rows2)
    sh = [r for r in rows2 if r["len_y"] <= med_len]
    lo = [r for r in rows2 if r["len_y"] > med_len]
    p(f"short/long bias (median len {med_len:.0f}): A per-byte {st.median(r['A'] for r in sh):.4f} vs "
      f"{st.median(r['A'] for r in lo):.4f}; A_fixed {st.median(r['A_fixed'] for r in sh):.4f} vs "
      f"{st.median(r['A_fixed'] for r in lo):.4f}")
    # Are repeat steps short? (if so, per-byte A was *rewarding* the loop pathology)
    rep_len = st.median(r["len_y"] for r in rows2 if r["repeat"])
    non_len = st.median(r["len_y"] for r in rows2 if not r["repeat"])
    p(f"repeat steps median len {rep_len:.0f} chars vs non-repeat {non_len:.0f}")
    p("")

    # ---------------- downside floor on A ----------------
    # A alone sees repeat steps but min(R,G,A) does not: A's large negatives
    # (refs disagree, the action matched none of the k modes) dominate the
    # min. max(A, −floor) bounds that cost; check what it does on both the
    # duel probe (paired z) and the bench steps (pathology discrimination).
    p("== downside floor: min(R, G, max(A_fixed, −floor)) ==")
    probe = json.load(open(REPO / "research/results/v6_action_leg_probe.json"))
    prow, ptau = probe["rows"], probe["tau"]

    def a_probe(side: str, r: dict) -> float:
        return lme([b * r[side]["len_y"] / L0 for b in r[side]["b"]], ptau)

    p(f"{'floor':>7} | duel probe {'margin':>9} {'SE':>8} {'z':>6} {'bindA':>6} | bench repeat {'z':>6} {'AUC':>6} | dead-tail {'z':>6} {'AUC':>6}")
    for f in (None, 0.05, 0.02, 0.01, 0.005, 0.0):
        def fl(a: float) -> float:
            return a if f is None else max(a, -f)
        ak = [fl(a_probe("king", r)) for r in prow]
        ac = [fl(a_probe("challenger", r)) for r in prow]
        dd = [min(r["challenger"]["R"], r["challenger"]["G"], c) - min(r["king"]["R"], r["king"]["G"], k)
              for r, k, c in zip(prow, ak, ac)]
        m = st.mean(dd); se = st.stdev(dd) / math.sqrt(len(dd))
        bind = sum(1 for r, k in zip(prow, ak) if k < min(r["king"]["R"], r["king"]["G"])) / len(prow)
        s = {id(r): min(r["R"], r["G"], fl(r["A_fixed"])) for r in rows2}
        rep = [s[id(r)] for r in rows2 if r["repeat"]]
        non = [s[id(r)] for r in rows2 if not r["repeat"]]
        dead = [s[id(r)] for r in rows2 if r["dead_tail"]]
        sub = [s[id(r)] for r in rows2 if r["submitted_run"] and not r["dead_tail"]]
        p(f"{str(f):>7} |            {m:+.5f} {se:.5f} {m / se:+.2f} {bind:6.0%} |              "
          f"{zstat(non, rep):6.2f} {auc(non, rep):6.3f} |           {zstat(sub, dead):6.2f} {auc(sub, dead):6.3f}")
    p("reference: min(R,G) alone on the duel probe z +3.53 (SE 0.00057); on bench repeat z +1.86.")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
