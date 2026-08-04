"""Analyze an E-LADDER / E-KINGS run: aggregate terms per miner, rank-correlate.

Usage: python scripts/analyze_ladder.py results/ladder_run1.jsonl [--truth qwen-ladder]
"""

import argparse
import json
from collections import defaultdict

from scipy import stats

# Known strength for the Qwen3 ladder: LiveCodeBench v5 (thinking mode),
# Qwen3 technical report. Any monotone-in-size proxy gives the same ranking.
TRUTHS = {
    "qwen-ladder": {
        "qwen3-1.7b": 33.2, "qwen3-4b": 54.2, "qwen3-8b": 57.5,
        "qwen3-14b": 63.5, "qwen3-32b": 65.7,
    },
    # swe-rebench (500-task) from albedo.tech/data/model-scores.json
    "kings": {
        "king-genesis": 58.2, "king-I": 38.4, "king-XCIX": 39.8,
        "king-VIII": 36.2, "king-XI": 33.6, "king-V": 32.0, "king-XLV": 26.0,
        "king-XLVI": 13.2, "king-LI": 11.6, "king-CI": 12.4,
    },
}

TERM_KEYS = ["D1", "D2", "D3", "D4", "D5"]
WEIGHTS = {"D1": 1.0, "D2": 1.0, "D3": 1.0, "D4": 0.5, "D5": 0.5}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("valid"):
                rows.append(r)
    return rows


def aggregate(rows, causality_min=0.0):
    per_miner = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["causality"] < causality_min:
            continue
        w = max(r["causality"], 0.0)
        for k in TERM_KEYS:
            per_miner[r["miner"]][k].append((r[k], w))
    out = {}
    for miner, terms in per_miner.items():
        agg = {}
        for k, vals in terms.items():
            agg[k] = sum(v for v, _ in vals) / len(vals)
            wsum = sum(w for _, w in vals)
            agg[k + "_cw"] = (sum(v * w for v, w in vals) / wsum) if wsum > 0 else agg[k]
        agg["n_turns"] = len(terms["D1"])
        # Score: lower loss = better miner (negated so higher S = better).
        agg["S"] = -sum(WEIGHTS[k] * agg[k] for k in TERM_KEYS)
        agg["S_cw"] = -sum(WEIGHTS[k] * agg[k + "_cw"] for k in TERM_KEYS)
        out[miner] = agg
    return out


def report(agg, truth):
    miners = [m for m in agg if m in truth]
    print(f"\n{'miner':<14}{'n':>5}" + "".join(f"{k:>9}" for k in TERM_KEYS)
          + f"{'S':>9}{'S_cw':>9}{'truth':>8}")
    for m in sorted(miners, key=lambda m: -truth[m]):
        a = agg[m]
        print(f"{m:<14}{a['n_turns']:>5}"
              + "".join(f"{a[k]:>9.4f}" for k in TERM_KEYS)
              + f"{a['S']:>9.4f}{a['S_cw']:>9.4f}{truth[m]:>8.1f}")
    if len(miners) >= 3:
        t = [truth[m] for m in miners]
        for sk in ["S", "S_cw"] + TERM_KEYS:
            s = [agg[m][sk] if sk.startswith("S") else -agg[m][sk] for m in miners]
            rho, p = stats.spearmanr(s, t)
            pr, pp = stats.pearsonr(s, t)
            print(f"  {sk:<6} vs truth: Spearman {rho:+.3f} (p={p:.3f})  "
                  f"Pearson {pr:+.3f} (p={pp:.3f})")


def ablations(rows, truth):
    """Spearman of S vs truth under alternative term weightings."""
    variants = {
        "default 1/1/1/.5/.5": WEIGHTS,
        "drop D2": {"D1": 1, "D2": 0, "D3": 1, "D4": 0.5, "D5": 0.5},
        "D1 only": {"D1": 1, "D2": 0, "D3": 0, "D4": 0, "D5": 0},
        "D1+D3": {"D1": 1, "D2": 0, "D3": 1, "D4": 0, "D5": 0},
        "capability only (D1+D3+D5)": {"D1": 1, "D2": 0, "D3": 1, "D4": 0, "D5": 1},
        "equal all": {k: 1.0 for k in TERM_KEYS},
    }
    agg = aggregate(rows)
    miners = [m for m in agg if m in truth]
    t = [truth[m] for m in miners]
    print("\n=== weight ablations (Spearman / Pearson of S vs truth) ===")
    for name, w in variants.items():
        s = [-sum(w[k] * agg[m][k] for k in TERM_KEYS) for m in miners]
        rho, p = stats.spearmanr(s, t)
        pr, _ = stats.pearsonr(s, t)
        print(f"  {name:<28} Spearman {rho:+.3f} (p={p:.3f})  Pearson {pr:+.3f}")


def pairwise_margins(rows, truth):
    """Paired per-turn S differences between adjacent-truth miners.

    Gives the win-margin calibration: how many turns are needed before the
    better model separates from the worse one at k standard errors (RT-4).
    """
    per = defaultdict(dict)  # turn_id -> miner -> S_turn
    for r in rows:
        per[r["turn_id"]][r["miner"]] = -sum(WEIGHTS[k] * r[k] for k in TERM_KEYS)
    miners = sorted({m for r in rows for m in [r["miner"]] if m in truth},
                    key=lambda m: -truth[m])
    print("\n=== paired margins (better vs next-worse, per-turn S diff) ===")
    for a, b in zip(miners, miners[1:]):
        diffs = [v[a] - v[b] for v in per.values() if a in v and b in v]
        if len(diffs) < 5:
            continue
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = (var / n) ** 0.5
        frac_win = sum(1 for d in diffs if d > 0) / n
        print(f"  {a:>14} vs {b:<14} n={n:<4} mean={mean:+.4f} se={se:.4f} "
              f"z={mean / se if se else 0:+.1f} turn-wins={frac_win:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--truth", default="qwen-ladder", choices=list(TRUTHS))
    ap.add_argument("--causality-min", type=float, default=None,
                    help="also report with turns filtered to causality >= this")
    args = ap.parse_args()
    rows = load(args.results)
    truth = TRUTHS[args.truth]
    print(f"{len(rows)} valid rows, "
          f"{len({r['turn_id'] for r in rows})} turns, "
          f"{len({r['miner'] for r in rows})} miners")
    print("\n=== all turns (causality-weighted columns use max(causality,0)) ===")
    report(aggregate(rows), truth)
    cmin = args.causality_min if args.causality_min is not None else 0.02
    print(f"\n=== causality-filtered turns (causality >= {cmin}) ===")
    report(aggregate(rows, causality_min=cmin), truth)
    ablations(rows, truth)
    pairwise_margins(rows, truth)


if __name__ == "__main__":
    main()
