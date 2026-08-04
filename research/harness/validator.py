"""Production duel validator: score king vs challenger, emit JSON verdict.

Thin drop-in CLI for the subnet duel pipeline. Reuses harness.duel scoring
loop and production S* math from harness.score — no duplicated formulas.

Optionally materializes a stratified duel slice via harness.sample_turns so
validators share a reproducible turn set (seed + n).

Usage (on pod):
  python -m harness.validator \
      --king king-genesis:8001 \
      --challenger king-I:8002 \
      --turns /root/data/turns_minicoder.jsonl --n-turns 80 --seed 7 \
      --ref-cache /root/results/ref_minicoder.jsonl \
      --out /root/results/duel_verdict.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from .duel import _json_safe, run_duel, verdict_from_summary
from .sample_turns import sample_turns


async def run(args):
    slice_path = None
    if args.materialize_slice:
        rows = [json.loads(l) for l in open(args.turns)]
        if args.stratify:
            picked = sample_turns(rows, args.n_turns, args.seed)
        else:
            rng = random.Random(args.seed)
            picked = list(rows)
            rng.shuffle(picked)
            picked = picked[: args.n_turns]
        slice_path = Path(args.out).with_suffix(".turns.jsonl")
        with open(slice_path, "w") as f:
            for r in picked:
                f.write(json.dumps(r) + "\n")
        args.n_turns = len(picked)
        args.turns = str(slice_path)
    summary, outp, rows_path = await run_duel(args)
    safe = _json_safe(summary)
    if slice_path is not None:
        safe["turn_slice"] = {
            "path": str(slice_path), "n": args.n_turns, "seed": args.seed,
            "stratify": args.stratify,
        }
    outp.write_text(json.dumps(safe, indent=2) + "\n")
    verdict = _json_safe(verdict_from_summary(summary))
    print(json.dumps(verdict), flush=True)
    print(f"wrote {outp} and {rows_path}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Run production duel and print JSON verdict on stdout")
    ap.add_argument("--king", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    ap.add_argument("--n-turns", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7,
                    help="RNG seed for stratified duel slice")
    ap.add_argument("--stratify", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--materialize-slice", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="write a seeded duel slice beside --out before scoring")
    ap.add_argument("--ref-cache", default="/root/results/ref_minicoder.jsonl")
    ap.add_argument("--out", required=True,
                    help="summary json path; per-turn rows go to .jsonl sibling")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--score-bank", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="compute per-pair L2_bank + miner bank_frac (default: on)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
