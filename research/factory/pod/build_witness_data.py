#!/usr/bin/env python3
"""Witness observation data: the environment as second teacher.

Every corpus turn whose prefix ends with (assistant action, user observation)
already contains a world-modeling example: given history + action, predict
what the environment said back. Emits companion files in the same format the
trainers already consume:
  --out-turns  turn entries  {turn_id: tid#obs, prefix: history + action}
  --out-rolls  rollout file  {turn_id: tid#obs, rollouts: [{raw: observation}]}
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gad_common import load_turn_meta, parse_prefix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/work/turns.jsonl.gz")
    ap.add_argument("--meta", default="/root/work/turn_meta.jsonl.gz")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-obs-chars", type=int, default=12000)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap emitted examples (held-out eval set)")
    ap.add_argument("--out-turns", required=True)
    ap.add_argument("--out-rolls", required=True)
    args = ap.parse_args()

    meta = load_turn_meta(args.meta)
    n = skip = 0
    with gzip.open(args.turns, "rt") as fh, \
         gzip.open(args.out_turns, "wt") as ot, open(args.out_rolls, "w") as orl:
        for line in fh:
            r = json.loads(line)
            tid = r["turn_id"]
            m = meta.get(tid)
            if not m or m.get("split") != args.split:
                continue
            msgs = parse_prefix(r.get("prefix"))
            if (not msgs or len(msgs) < 3 or msgs[-1].get("role") != "user"
                    or msgs[-2].get("role") != "assistant"):
                skip += 1
                continue
            obs = (msgs[-1].get("content") or "").strip()
            if not obs:
                skip += 1
                continue
            wid = tid + "#obs"
            ot.write(json.dumps({"turn_id": wid,
                                 "prefix": json.dumps(msgs[:-1])}) + "\n")
            orl.write(json.dumps({"turn_id": wid, "rollouts": [
                {"raw": obs[:args.max_obs_chars]}]}) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"DONE witness obs examples={n} skipped={skip}")


if __name__ == "__main__":
    main()
