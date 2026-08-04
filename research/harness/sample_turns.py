"""Stratified turn sampling for subnet duels.

Draws n turns from a pooled unlabeled jsonl, optionally stratified by a
coarse phase tag inferred from traj_id / metadata so duels are not dominated
by one repo or bug family.

Usage:
  python -m harness.sample_turns \
      --turns /root/data/turns_minicoder.jsonl --n 80 --seed 7 \
      --out /root/data/duel_slice.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


_PHASE_RE = re.compile(r"(func_basic|func_pm|lm_rewrite|lm_modify|combine_|pr_\d+)")


def phase_key(rec: dict) -> str:
    """Coarse stratum: repo family × phase tag (best-effort)."""
    tid = rec.get("traj_id", "")
    repo = tid.split(".", 1)[0] if tid else "unknown"
    m = _PHASE_RE.search(tid)
    phase = m.group(1) if m else "other"
    return f"{repo}|{phase}"


def sample_turns(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample without replacement; falls back to flat shuffle."""
    if n >= len(rows):
        rng = random.Random(seed)
        out = list(rows)
        rng.shuffle(out)
        return out
    by = defaultdict(list)
    for r in rows:
        by[phase_key(r)].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by)
    # round-robin until n
    out: list[dict] = []
    idx = {k: 0 for k in keys}
    while len(out) < n:
        progress = False
        for k in keys:
            i = idx[k]
            if i < len(by[k]):
                out.append(by[k][i])
                idx[k] = i + 1
                progress = True
                if len(out) >= n:
                    break
        if not progress:
            break
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.turns)]
    if args.stratify:
        picked = sample_turns(rows, args.n, args.seed)
    else:
        rng = random.Random(args.seed)
        picked = list(rows)
        rng.shuffle(picked)
        picked = picked[: args.n]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    strata = defaultdict(int)
    for r in picked:
        strata[phase_key(r)] += 1
    print(json.dumps({
        "n": len(picked), "seed": args.seed, "stratify": args.stratify,
        "n_strata": len(strata),
        "top_strata": sorted(strata.items(), key=lambda x: -x[1])[:10],
        "out": str(outp),
    }))


if __name__ == "__main__":
    main()
