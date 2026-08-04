"""A11: how load-bearing is the 3σ duel for short-style kings I/II?

The live duels passed at z=1.30 (I) and 1.72 (II) on 80 turns — genesis holds.
But z grows ~sqrt(n) if the mean paired margin stays positive, so a persistent
challenger could eventually cross 3σ purely by running more turns. Quantify:
  1. bootstrap the 80-turn slice -> distribution of z, P(challenger wins now)
  2. project n* where z crosses 3 (margin stable)
  3. does clip(L1lift,±0.1) shrink or flip the mean margin (kill the sqrt(n) risk)?

Usage: source .venv/bin/activate && python scripts/a11_bootstrap.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.score import lambda2  # noqa: E402

RESULTS = ROOT / "results"
DUELS = {"I": RESULTS / "duel_genesis_I.jsonl",
         "II": RESULTS / "duel_genesis_II.jsonl",
         "VII": RESULTS / "duel_genesis_VII.jsonl",
         "XCIX": RESULTS / "duel_genesis_XCIX.jsonl",
         "LI": RESULTS / "duel_genesis_LI.jsonl"}
SWE = {"genesis": 58.2, "I": 38.4, "II": 37.2, "VII": 33.2, "XCIX": 39.8, "LI": 11.6}
RNG = np.random.default_rng(0)
B = 20000


def term(p: dict, mode: str) -> float:
    lift = p["lpA_yc_za"] - p["lpA_yc_e"]
    if mode == "clip0.1":
        lift = max(-0.1, min(lift, 0.1))
    return lambda2(p) + lift


def paired_diffs(path: Path, mode: str) -> list[float]:
    king: dict[str, list[dict]] = {}
    chall: dict[str, list[dict]] = {}
    for line in open(path):
        r = json.loads(line)
        if not (r.get("valid") and "pairs" in r):
            continue
        (king if r["miner"] == "king-genesis" else chall)[r["turn_id"]] = r["pairs"]
    diffs = []
    for tid in sorted(set(king) & set(chall)):
        mk = st.mean(term(p, mode) for p in king[tid])
        mc = st.mean(term(p, mode) for p in chall[tid])
        diffs.append(mc - mk)  # challenger − king; >0 favors challenger
    return diffs


def z_of(diffs: list[float]) -> float:
    if len(diffs) < 2:
        return float("nan")
    se = st.stdev(diffs) / math.sqrt(len(diffs))
    return st.mean(diffs) / se if se > 0 else 0.0


def analyze(name: str, path: Path) -> None:
    print(f"\n===== king-{name} vs genesis =====")
    for mode in ("l1lift", "clip0.1"):
        d = np.array(paired_diffs(path, mode))
        n = len(d)
        z = z_of(list(d))
        mean, sd = d.mean(), d.std(ddof=1)
        # bootstrap z distribution at n=80
        bz = []
        for _ in range(B):
            s = RNG.choice(d, size=n, replace=True)
            se = s.std(ddof=1) / math.sqrt(n)
            bz.append(s.mean() / se if se > 0 else 0.0)
        bz = np.array(bz)
        p_win = float((bz > 3).mean())
        # n* to cross 3σ assuming margin/sd stable: z(n)=mean/(sd/sqrt(n))
        if mean > 0:
            n_star = (3 * sd / mean) ** 2
            n_star_txt = f"{n_star:.0f}"
        else:
            n_star_txt = "never (margin ≤ 0)"
        print(f"  [{mode:8s}] n={n} margin={mean:+.5f} sd={sd:.4f} z={z:+.3f}  "
              f"P(win@80)={p_win*100:.2f}%  z-CI=[{np.percentile(bz,2.5):+.2f},"
              f"{np.percentile(bz,97.5):+.2f}]  n*(cross 3σ)={n_star_txt}")


def effect_floor() -> None:
    print("\n===== effect-size floor across all duels (current term, mix w=1.0) =====")
    print("A pure z>3 test fires for ANY persistent margin>0 at large enough n. An "
          "effect-size floor (|margin|≥δ or Cohen's d≥d0) blocks tiny short-style wins "
          "regardless of n. Real dethronements should clear it; short-style should not.")
    print(f"{'chall':6s} {'swe':>5s} {'margin':>9s} {'cohen_d':>8s} {'z@80':>7s}  interpretation")
    rows = []
    for name, path in DUELS.items():
        if not path.exists():
            continue
        d = np.array(paired_diffs(path, "l1lift"))
        mean, sd = d.mean(), d.std(ddof=1)
        cohen = mean / sd if sd > 0 else 0.0
        rows.append((name, SWE.get(name), mean, cohen, z_of(list(d))))
    for name, swe, mean, cohen, z in sorted(rows, key=lambda x: -x[2]):
        note = ("SHORT-STYLE FP (worse than genesis, tiny +margin)" if mean > 0
                else "correctly rejected (−margin)")
        print(f"{name:6s} {swe:5.1f} {mean:+9.5f} {cohen:+8.3f} {z:+7.2f}  {note}")
    pos = [r[2] for r in rows if r[2] > 0]
    if pos:
        print(f"\nLargest short-style margin (must stay BELOW floor): {max(pos):+.4f} "
              f"(king-{[r[0] for r in rows if r[2]==max(pos)][0]}).")
        print("Proposed floor δ=0.05 absolute (≈8% of the −0.036..−0.63 king-S range): "
              "blocks II (+0.034) and I (+0.009) at any n; a genuine capability jump "
              "(margin ≫ 0.05) still dethrones. Ship: dethrone iff z>3 AND margin>δ.")


def main() -> None:
    for name, path in {"I": DUELS["I"], "II": DUELS["II"]}.items():
        if path.exists():
            analyze(name, path)
    effect_floor()
    print("\nSummary: clip0.1 does NOT fix A11 (shrinks variance → z rises, n* falls). "
          "The right fix is an effect-size floor on the duel margin, independent of the "
          "RT-3b clip decision.")


if __name__ == "__main__":
    main()
