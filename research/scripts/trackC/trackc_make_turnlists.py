#!/usr/bin/env python3
"""Pick the fixed turn lists for Track C, once, seeded.

train_pool.json  - train-split turns the generator loop samples from and the
                   teacher/G0 rollouts cover (phases 0-2).
heldout.json     - test-split turns (different repos than train, by the corpus
                   split) used for D held-out accuracy, fool rate, and action
                   agreement. Never trained on by G or D.
"""
from __future__ import annotations

import argparse
import json
import os
import random

from trackc_common import load_splits, load_turns, parse_prefix


def usable(rec, max_chars):
    msgs = parse_prefix(rec.get("prefix"))
    if not msgs:
        return False
    total = sum(len(m.get("content") or "") for m in msgs)
    return 0 < total <= max_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/work/research/data/disc_pairs")
    ap.add_argument("--train-n", type=int, default=5000)
    ap.add_argument("--heldout-n", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--out", default="/root/work/trackC")
    args = ap.parse_args()

    turns = load_turns(os.path.join(args.data, "turns.jsonl.gz"))
    splits = load_splits(args.data)
    rng = random.Random(args.seed)

    pools = {"train": [], "test": []}
    for tid in sorted(turns):
        sp = splits.get(tid)
        if sp in pools and usable(turns[tid], args.max_chars):
            pools[sp].append(tid)
    for v in pools.values():
        rng.shuffle(v)

    train_pool = pools["train"][: args.train_n]
    heldout = pools["test"][: args.heldout_n]
    os.makedirs(args.out, exist_ok=True)
    json.dump(train_pool, open(os.path.join(args.out, "train_pool.json"), "w"))
    json.dump(heldout, open(os.path.join(args.out, "heldout.json"), "w"))
    print(f"train candidates={len(pools['train'])} -> pool={len(train_pool)}")
    print(f"test candidates={len(pools['test'])} -> heldout={len(heldout)}")


if __name__ == "__main__":
    main()
