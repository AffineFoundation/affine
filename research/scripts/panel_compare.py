#!/usr/bin/env python3
"""Compare panel runs on their shared instances, with a paired test.

Headline resolve rates are a poor way to compare two models on 150 instances:
most instances are either solved by everyone or by no one, and those contribute
nothing but noise. McNemar's test looks only at the instances where the two runs
disagree, which is where the information actually is.

usage:
  panel_compare.py --baseline results/gen_full.json \
                   --runs results/g_round0005.json results/g_round0020.json \
                   --teacher results/teacher_full.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    return d, {k: bool(v) for k, v in (d.get("per_instance") or {}).items()}


def mcnemar(a, b):
    """a, b: instance -> resolved. Returns (n_shared, a_only, b_only, p_two_sided).

    Exact binomial test on the discordant pairs, which is valid at the small
    discordant counts this panel produces.
    """
    shared = sorted(set(a) & set(b))
    a_only = sum(1 for k in shared if a[k] and not b[k])
    b_only = sum(1 for k in shared if b[k] and not a[k])
    n = a_only + b_only
    if n == 0:
        return len(shared), 0, 0, 1.0
    lo = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return len(shared), a_only, b_only, min(1.0, 2 * tail)


def rate(d, keys=None):
    items = d if keys is None else {k: d[k] for k in keys if k in d}
    n = len(items)
    r = sum(1 for v in items.values() if v)
    return r, n, (r / n if n else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="generator baseline run")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--glob", default=None,
                    help="e.g. 'results/g_round*.json' to pick up all checkpoints")
    args = ap.parse_args()

    base_d, base = load(args.baseline)
    r, n, pct = rate(base)
    print(f"baseline  {os.path.basename(args.baseline):26s} "
          f"{r:3d}/{n:3d} = {pct:6.2%}")

    tea = None
    if args.teacher and os.path.exists(args.teacher):
        tea_d, tea = load(args.teacher)
        r, n, pct = rate(tea)
        print(f"teacher   {os.path.basename(args.teacher):26s} "
              f"{r:3d}/{n:3d} = {pct:6.2%}")
        ns, ao, bo, p = mcnemar(base, tea)
        print(f"          teacher vs baseline: shared={ns} "
              f"base_only={ao} teacher_only={bo} p={p:.4f}")
        print(f"          -> gap to close = "
              f"{rate(tea)[2] - rate(base)[2]:+.2%}")

    runs = list(args.runs)
    if args.glob:
        runs += sorted(glob.glob(args.glob))
    runs = [x for x in dict.fromkeys(runs) if os.path.exists(x)]
    if not runs:
        return

    print("\ntrajectory (each vs the generator baseline, paired):")
    print(f"{'run':26s} {'resolved':>10s} {'rate':>8s} {'d_vs_base':>10s} "
          f"{'base_only':>9s} {'ckpt_only':>9s} {'p':>7s}  frac_of_gap")
    for path in runs:
        _, cur = load(path)
        r, n, pct = rate(cur)
        ns, ao, bo, p = mcnemar(base, cur)
        _, _, bpct = rate(base, cur.keys())
        d = pct - bpct
        frac = ""
        if tea:
            _, _, tpct = rate(tea, cur.keys())
            gap = tpct - bpct
            if abs(gap) > 1e-9:
                frac = f"{d / gap:+.0%}"
        print(f"{os.path.basename(path):26s} {r:4d}/{n:<5d} {pct:7.2%} "
              f"{d:+9.2%} {ao:9d} {bo:9d} {p:7.4f}  {frac}")


if __name__ == "__main__":
    main()
