#!/usr/bin/env python3
"""Merge sharded A/B eval dumps and recompute every metric and control.

Each shard writes per-pair records carrying p_ref plus both candidate lengths,
so the merged file is self-sufficient: accuracy, the length bar and the
length-matched slice are all recomputed here rather than averaged from shards
(averaging ratios across unequal shards would be wrong).
"""
from __future__ import annotations

import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="e.g. '/root/work/zs/zeroshot_scores.shard*.jsonl'")
    ap.add_argument("--label", default="merged")
    args = ap.parse_args()

    recs = []
    files = []
    for g in args.globs:
        for f in sorted(glob.glob(g)):
            files.append(f)
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
    # a pair could appear once per shard only; guard against double counting
    seen = {}
    for r in recs:
        seen[(r["repo"], r["turn_id"])] = r
    recs = list(seen.values())

    if not recs:
        raise SystemExit("no records matched")

    n = len(recs)
    correct = sum(1 for r in recs if r["p_ref"] > 0.5)
    lc = sum(1 for r in recs if r["len_ref"] > r["len_mine"])
    lt = sum(1 for r in recs if r["len_ref"] == r["len_mine"])
    lacc = (lc + 0.5 * lt) / n
    matched = [r for r in recs
               if abs(r["len_ref"] - r["len_mine"]) / (max(r["len_ref"], r["len_mine"]) or 1) <= 0.10]
    macc = (sum(1 for r in matched if r["p_ref"] > 0.5) / len(matched)) if matched else float("nan")

    acc = correct / n
    lbar = max(lacc, 1 - lacc)
    print(f"[{args.label}] shards={len(files)}  n={n}")
    print(f"  accuracy          = {acc:.4f}")
    print(f"  length bar        = {lbar:.4f}   (raw 'teacher longer' = {lacc:.4f})")
    print(f"  length-matched    = {macc:.4f}   (n={len(matched)})")
    print(f"  mean p(teacher)   = {sum(r['p_ref'] for r in recs)/n:.4f}")
    print()
    print("  verdict: " + (
        "beats the length bar" if acc > lbar + 0.02 else
        "AT OR BELOW the length bar -- no signal beyond verbosity"))


if __name__ == "__main__":
    main()
