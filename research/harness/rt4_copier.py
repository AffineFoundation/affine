"""RT-4: same-model paired variance — calibrate the win margin δ.

Scores the SAME miner twice on identical teacher refs (two independent rollout
sets). The paired difference distribution is the null for a perfect king-copier;
dethronement requires beating that null by k·SE.

Usage:
  python -m harness.rt4_copier --miner king-XLVI=...:8002 --n-turns 100 \
      --out /root/results/rt4_copier.jsonl
"""

import argparse
import asyncio
import json
import pathlib

import httpx

from .client import VllmModel
from .config import TEACHER, ModelCfg, RunCfg
from .runner import load_turns, turn_id
from .terms import miner_terms, teacher_reference


async def run(args):
    rc = RunCfg()
    sem = asyncio.Semaphore(rc.max_concurrency)
    name, rest = args.miner.split("=", 1)
    repo, port = rest.rsplit(":", 1)
    async with httpx.AsyncClient() as http:
        teacher = VllmModel(TEACHER, http, sem)
        miner = VllmModel(
            ModelCfg(name=name, repo=repo, port=int(port), family="generic",
                     gpus="?"), http, sem)

        turns = load_turns(args.turns, args.n_turns)
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out_f = open(outp, "a")

        ref_cache = {}
        if args.ref_cache and pathlib.Path(args.ref_cache).exists():
            with open(args.ref_cache) as f:
                for line in f:
                    r = json.loads(line)
                    ref_cache[r["turn_id"]] = r["ref"]

        for ti, rec in enumerate(turns):
            tid = turn_id(rec)
            prefix = rec["prefix"]
            ref = ref_cache.get(tid)
            if ref is None:
                ref = await teacher_reference(
                    teacher, prefix, rc.n_teacher_samples, rc.temperature,
                    rc.max_thought_tokens, rc.max_action_tokens)
            if not ref:
                continue
            causality = sum(r["lp_own"] - r["lp_empty"] for r in ref) / len(ref)

            # Two independent miner rollout sets against the same teacher refs.
            a = await miner_terms(teacher, miner, prefix, ref, rc.n_miner_samples,
                                  rc.temperature, rc.max_thought_tokens,
                                  rc.max_action_tokens)
            b = await miner_terms(teacher, miner, prefix, ref, rc.n_miner_samples,
                                  rc.temperature, rc.max_thought_tokens,
                                  rc.max_action_tokens)
            row = {"turn_id": tid, "causality": causality, "A": a, "B": b}
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            print(f"[{ti + 1}/{len(turns)}] {tid}", flush=True)
        out_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    ap.add_argument("--miner", required=True)
    ap.add_argument("--n-turns", type=int, default=100)
    ap.add_argument("--ref-cache", default="/root/results/ref_minicoder.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
