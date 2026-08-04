"""Calibrate win-margin δ from E-KINGS paired L2 differences + RT-4 null.

Usage:
  python scripts/calibrate_delta.py results/ekings_v2_all.jsonl \
      [--rt4 results/rt4_copier.jsonl]
"""

import argparse
import json
import math
import statistics as st
from collections import defaultdict

from scipy import stats

TRUTH = {
    "king-genesis": 58.2, "king-I": 38.4, "king-XCIX": 39.8, "king-VIII": 36.2,
    "king-XI": 33.6, "king-V": 32.0, "king-XLV": 26.0, "king-XLVI": 13.2,
    "king-LI": 11.6, "king-CI": 12.4,
}


def l2_row(r):
    return st.mean(p["lpC_yc_za"] - p["lpC_yc_e"] for p in r["pairs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--rt4", default=None)
    ap.add_argument("--k", type=float, default=3.0)
    args = ap.parse_args()

    by = defaultdict(dict)
    for line in open(args.results):
        r = json.loads(line)
        if r.get("valid") and "pairs" in r and r["miner"] in TRUTH:
            by[r["turn_id"]][r["miner"]] = r

    miners = sorted(TRUTH, key=lambda m: -TRUTH[m])
    print(f"{'pair':<32}{'n':>5}{'mean':>10}{'se':>10}{'z':>8}"
          f"{'3σ_win?':>10}{'turns@3σ':>10}")
    for a, b in zip(miners, miners[1:]):
        diffs = []
        for tid, d in by.items():
            if a in d and b in d:
                diffs.append(l2_row(d[a]) - l2_row(d[b]))
        n = len(diffs)
        mean = st.mean(diffs)
        se = st.stdev(diffs) / math.sqrt(n)
        z = mean / se
        # turns needed for |mean| to exceed k·se assuming same per-turn var
        var1 = st.variance(diffs)
        n_need = math.ceil((args.k ** 2) * var1 / (mean ** 2)) if mean else float("inf")
        print(f"{a.removeprefix('king-')+' vs '+b.removeprefix('king-'):<32}"
              f"{n:>5}{mean:>+10.5f}{se:>10.5f}{z:>+8.1f}"
              f"{'YES' if abs(z) >= args.k else 'no':>10}{n_need:>10}")

    # Overall: how often would a worse-bench king wrongly dethrone a better one
    # under k·SE rule?
    wrong = right = ties = 0
    for a in miners:
        for b in miners:
            if TRUTH[a] <= TRUTH[b]:
                continue
            diffs = [l2_row(d[a]) - l2_row(d[b])
                     for d in by.values() if a in d and b in d]
            if len(diffs) < 2:
                continue
            mean = st.mean(diffs)
            se = st.stdev(diffs) / math.sqrt(len(diffs))
            if mean > args.k * se:
                right += 1
            elif mean < -args.k * se:
                wrong += 1  # worse model would dethrone better
            else:
                ties += 1
    print(f"\nAmong pairs where bench(A)>bench(B): A dethrones B at {args.k}σ: "
          f"{right}; B wrongly dethrones A: {wrong}; inconclusive: {ties}")

    if args.rt4:
        null = []
        for line in open(args.rt4):
            r = json.loads(line)
            if not (r.get("A", {}).get("valid") and r.get("B", {}).get("valid")):
                continue
            if "pairs" not in r["A"] or "pairs" not in r["B"]:
                # fall back to D2 if v1-only
                if "D2" in r["A"]:
                    # D2 = lpC_yc_zc - lpC_yc_za ⇒ L2 ≈ causality - D2, cancel
                    null.append(-(r["A"]["D2"] - r["B"]["D2"]))
                continue
            la = st.mean(p["lpC_yc_za"] - p["lpC_yc_e"] for p in r["A"]["pairs"])
            lb = st.mean(p["lpC_yc_za"] - p["lpC_yc_e"] for p in r["B"]["pairs"])
            null.append(la - lb)
        if null:
            print(f"\nRT-4 same-model null: n={len(null)} mean={st.mean(null):+.5f} "
                  f"sd={st.stdev(null):.5f} "
                  f"P(|diff|>0 at 3σ with n={len(null)}) ≈ "
                  f"{2 * (1 - stats.norm.cdf(args.k)):.4f} (should be ~0.0027)")


if __name__ == "__main__":
    main()
