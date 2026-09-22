"""Offline re-normalization of the v6 A-leg probe (2026-09-10).

Why: the 2026-09-04 probe (v6_action_leg_probe.py, 1,222 turns of
chal-00248) built the A leg per byte of the action and found it paid short
actions ~10x (king short 0.021 vs long 0.002) and collapsed the crown z
+3.5 → +0.5. This script re-normalizes the SAME echoes without a GPU: the
stored rows carry per-byte b_i and len_y for every side, so the summed
lift S_i = b_i · len_y is recoverable, and every normalization that is a
function of (S_i, len) can be compared on identical data.

Candidates:
  perbyte   b_i                              (as built)
  fixed     S_i / L0                         (staged: [duel].action_norm_bytes)
  fence     (S_i − c) / (len − 12)           (body-only proxy; c = boilerplate lift)
  lenresid  b_i − f(len)                     (subtract teacher-own length trend)
  relative  LME(b over refs 2,3) − LME(teacher_own b)   (vs the teacher's own action)

Headline: on the teacher's own actions S = 0.61 + 0.00007·len (Spearman
+0.05) — the lift a thought gives an action is a near-constant ~0.6 nats
that does not grow with length. Dividing by the real length is therefore
the bias itself; dividing by a fixed L0 removes it (ratio 0.77 at L0=128)
and restores the crown signal (z +2.32; +2.93 at L0=192).

    python research/scripts/v6_action_leg_norm.py \
        research/results/v6_action_leg_probe.json > research/results/v6_action_leg_norm.txt
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys

import numpy as np

GENERIC_ACTION = "```bash\nls -la\n```"
FENCE_BYTES = len("```bash\n") + len("\n```")


def lme(vals: list[float], tau: float | None) -> float:
    if len(vals) == 1 or tau is None or tau <= 0:
        return st.mean(vals)
    m = max(vals)
    return m + tau * math.log(st.mean(math.exp((v - m) / tau) for v in vals))


def med(xs: list[float]) -> float:
    return st.median(xs)


def side_len(row: dict, side: str) -> int:
    return row[side]["len_y"] if side != "generic" else len(GENERIC_ACTION)


def paired(rows: list[dict], a_king: list[float], a_chal: list[float]) -> tuple[float, float, float, dict]:
    dd = [min(r["challenger"]["R"], r["challenger"]["G"], c) - min(r["king"]["R"], r["king"]["G"], k)
          for r, k, c in zip(rows, a_king, a_chal)]
    m = st.mean(dd)
    se = st.stdev(dd) / math.sqrt(len(dd))
    binds = {"R": 0, "G": 0, "A": 0}
    for r, k in zip(rows, a_king):
        legs = {"R": r["king"]["R"], "G": r["king"]["G"], "A": k}
        binds[min(legs, key=legs.get)] += 1
    n = len(rows)
    return m, se, m / se, {k: v / n for k, v in binds.items()}


def main() -> int:
    d = json.load(open(sys.argv[1]))
    rows, tau = d["rows"], d["tau"]
    n = len(rows)
    out: list[str] = []
    p = out.append

    p(f"v6 A-leg re-normalization — {n} turns of {d['record']} (tau={tau})")
    p("")
    # --- how does the summed lift depend on length? (teacher's own actions)
    L = np.array([r["teacher_own"]["len_y"] for r in rows for _ in r["teacher_own"]["b"]], float)
    S = np.array([b * r["teacher_own"]["len_y"] for r in rows for b in r["teacher_own"]["b"]])
    beta, c = np.polyfit(L, S, 1)
    rho = np.corrcoef(np.argsort(np.argsort(S)), np.argsort(np.argsort(L)))[0, 1]
    p("teacher_own per-ref summed lift S = Σ_bytes[lpC(y|z_C) − lpC(y|∅)]:")
    p(f"  S = {c:+.3f} + {beta:+.5f}·len   (n={len(S)}; Spearman(S, len) = {rho:+.3f})")
    p(f"  median S {np.median(S):+.3f}, p10 {np.percentile(S, 10):+.3f}, p90 {np.percentile(S, 90):+.3f}")
    p("  → the lift is a near-constant per action; per-real-byte division is the length bias.")
    p("")

    # --- min(R,G) baseline
    d_rg = [min(r["challenger"]["R"], r["challenger"]["G"]) - min(r["king"]["R"], r["king"]["G"]) for r in rows]
    m = st.mean(d_rg); se = st.stdev(d_rg) / math.sqrt(n)
    p(f"paired margin under min(R,G)        : {m:+.5f}  SE {se:.5f}  z {m / se:+.2f}")
    p("")

    # --- fixed-byte sweep
    p("fixed-byte normalization  A = LME_tau(S_i / L0)")
    p(f"{'L0':>5} {'side':12} {'A med':>8} {'A<0':>6} {'short':>8} {'long':>8} {'ratio':>6} {'>gen':>6} "
      f"| {'margin':>9} {'SE':>8} {'z':>6} {'bind R/G/A':>12}")
    for L0 in (64, 96, 128, 160, 192, 256, 384):
        def a_fixed(side: str, r: dict) -> float:
            return lme([b * side_len(r, side) / L0 for b in r[side]["b"]], tau)
        ak = [a_fixed("king", r) for r in rows]
        ac = [a_fixed("challenger", r) for r in rows]
        gen = [a_fixed("generic", r) for r in rows]
        for side, vals in (("king", ak), ("challenger", ac),
                           ("teacher_own", [a_fixed("teacher_own", r) for r in rows])):
            lmed = med([r[side]["len_y"] for r in rows])
            sh = [a for a, r in zip(vals, rows) if r[side]["len_y"] <= lmed]
            lo = [a for a, r in zip(vals, rows) if r[side]["len_y"] > lmed]
            line = (f"{L0:5d} {side:12} {med(vals):8.4f} {sum(v < 0 for v in vals) / n:6.1%} "
                    f"{med(sh):8.4f} {med(lo):8.4f} {med(sh) / med(lo):6.2f} "
                    f"{sum(a > g for a, g in zip(vals, gen)) / n:6.1%}")
            if side == "king":
                mm, ss, z, binds = paired(rows, ak, ac)
                line += (f" | {mm:+.5f} {ss:.5f} {z:+.2f}   "
                         f"{binds['R']:.0%}/{binds['G']:.0%}/{binds['A']:.0%}")
            p(line)
        p("")

    # --- the other candidates, at a glance
    p("other candidates (king side; ratio = A med short / A med long; z = paired min(R,G,A))")
    c_robust = float(np.median(S[L <= np.percentile(L, 20)]))
    A_t = np.array([lme(r["teacher_own"]["b"], tau) for r in rows])
    trend = np.polyfit(np.log([r["teacher_own"]["len_y"] for r in rows]), A_t, 1)

    def cand(mode: str, side: str, r: dict) -> float:
        b, ln = r[side]["b"], side_len(r, side)
        if mode == "perbyte":
            return lme(b, tau)
        if mode == "fence":
            return lme([(x * ln - c_robust) / max(ln - FENCE_BYTES, 1) for x in b], tau)
        if mode == "lenresid":
            return lme(b, tau) - float(np.polyval(trend, math.log(ln)))
        if mode == "relative":
            if side == "teacher_own":
                return 0.0
            return lme(b[1:], tau) - lme(r["teacher_own"]["b"], tau)
        raise ValueError(mode)

    p(f"{'mode':9} {'A med':>8} {'A<0':>6} {'own<0':>6} {'ratio':>6} {'>gen':>6} | {'margin':>9} {'SE':>8} {'z':>6} {'bindA':>6}")
    for mode in ("perbyte", "fence", "lenresid", "relative"):
        ak = [cand(mode, "king", r) for r in rows]
        ac = [cand(mode, "challenger", r) for r in rows]
        own = [cand(mode, "teacher_own", r) for r in rows]
        gen = [cand(mode, "generic", r) for r in rows]
        lmed = med([r["king"]["len_y"] for r in rows])
        sh = [a for a, r in zip(ak, rows) if r["king"]["len_y"] <= lmed]
        lo = [a for a, r in zip(ak, rows) if r["king"]["len_y"] > lmed]
        ratio = med(sh) / med(lo) if med(lo) else float("nan")
        mm, ss, z, binds = paired(rows, ak, ac)
        p(f"{mode:9} {med(ak):8.4f} {sum(v < 0 for v in ak) / n:6.1%} {sum(v < 0 for v in own) / n:6.1%} "
          f"{ratio:6.2f} {sum(a > g for a, g in zip(ak, gen)) / n:6.1%} | {mm:+.5f} {ss:.5f} {z:+.2f} {binds['A']:6.0%}")
    p("")
    p("notes: `fence` subtracts the boilerplate constant then divides by body length — worse,")
    p("because the constant IS the signal; `relative` centers on the teacher's own action so A")
    p("binds >50% and uses 2 refs (noisier); `lenresid` centers to ~0 and binds >50%.")
    p("teacher-own A<0 (~30%) persists under every form: refs disagree by a median 1.5 nats,")
    p("so a valid action matching a fourth mode scores negative — symmetric across sides.")

    # --- tau sensitivity at L0=128
    p("")
    p("tau sensitivity at L0=128 (contract tau is 0.03)")
    p(f"{'tau':>6} {'own<0':>6} {'king<0':>7} | {'margin':>9} {'SE':>8} {'z':>6} {'bindA':>6}")
    for t in (0.01, 0.03, 0.06, 0.1, None):
        def a_t(side: str, r: dict) -> float:
            return lme([b * side_len(r, side) / 128.0 for b in r[side]["b"]], t)
        ak = [a_t("king", r) for r in rows]; ac = [a_t("challenger", r) for r in rows]
        own = [a_t("teacher_own", r) for r in rows]
        mm, ss, z, binds = paired(rows, ak, ac)
        p(f"{str(t):>6} {sum(v < 0 for v in own) / n:6.1%} {sum(v < 0 for v in ak) / n:7.1%} "
          f"| {mm:+.5f} {ss:.5f} {z:+.2f} {binds['A']:6.0%}")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
