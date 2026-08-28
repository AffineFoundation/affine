#!/usr/bin/env python3
"""Paired comparison of SWE bench runs (McNemar exact test).

Why paired: with ~150 instances, an unpaired 28% -> 35% move is ~1.3 SE and
unresolvable. Pairing on identical instances discards the tasks that are
hopeless-for-both or trivial-for-both and tests only the discordant ones,
which is the sensitive and correct comparison for "did this checkpoint get
better than the baseline".

Inputs are per-instance artifacts persisted by the bench runners, in either
form:
  {"instances": {"<iid>": {"resolved": true, ...}, ...}, ...}   (swerunner _artifact)
  {"<iid>": true/false, ...}                                    (flat map)

Usage:
  analyze_paired_bench.py BASELINE.json OTHER.json [OTHER2.json ...]
Each OTHER is compared against BASELINE on the intersection of instances.
"""
from __future__ import annotations

import json
import math
import sys


def load(path):
    d = json.load(open(path))
    if isinstance(d, dict) and isinstance(d.get("instances"), dict):
        return {k: bool(v.get("resolved")) for k, v in d["instances"].items()}
    if isinstance(d, dict) and isinstance(d.get("result"), dict) \
            and isinstance(d.get("instances"), dict):  # job wrapper
        return {k: bool(v.get("resolved")) for k, v in d["instances"].items()}
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, dict) and "resolved" in v:
                out[k] = bool(v["resolved"])
        if out:
            return out
    raise SystemExit(f"unrecognized artifact format: {path}")


def mcnemar_exact(b, c):
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    base_path, others = sys.argv[1], sys.argv[2:]
    base = load(base_path)
    print(f"baseline: {base_path}  n={len(base)}  "
          f"resolved={sum(base.values())} ({sum(base.values())/len(base):.1%})")
    for op in others:
        o = load(op)
        common = sorted(set(base) & set(o))
        if not common:
            print(f"\n{op}: NO OVERLAPPING INSTANCES - cannot pair")
            continue
        both = sum(1 for i in common if base[i] and o[i])
        neither = sum(1 for i in common if not base[i] and not o[i])
        only_base = sum(1 for i in common if base[i] and not o[i])
        only_other = sum(1 for i in common if not base[i] and o[i])
        p = mcnemar_exact(only_base, only_other)
        rb = sum(base[i] for i in common) / len(common)
        ro = sum(o[i] for i in common) / len(common)
        print(f"\n{op}")
        print(f"  paired n={len(common)}  baseline {rb:.1%} -> this {ro:.1%}")
        print(f"  both solved={both}  neither={neither}  "
              f"only-baseline={only_base}  only-this={only_other}")
        print(f"  McNemar exact p = {p:.4f}"
              + ("   ** significant at 0.05" if p < 0.05 else ""))
        gained = [i for i in common if o[i] and not base[i]]
        lost = [i for i in common if base[i] and not o[i]]
        if gained:
            print(f"  gained: {', '.join(gained[:6])}" + (" ..." if len(gained) > 6 else ""))
        if lost:
            print(f"  lost  : {', '.join(lost[:6])}" + (" ..." if len(lost) > 6 else ""))


if __name__ == "__main__":
    main()
