"""Re-score all miners in a v2 jsonl under Λ2_bank using their recorded z_a.

Teacher-only. For each (turn, miner, pair) computes
  L2_bank = lpC(y_C|z_A) − max_k lpC(y_C|prior_k)
and writes a compact results file for offline Spearman.

Usage:
  python -m harness.rescore_bank --src /root/results/ekings_w1_v2.jsonl \
      --out /root/results/bank_w1.jsonl --n-turns 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from collections import defaultdict

import httpx

from .client import VllmModel
from .config import TEACHER, RunCfg
from .priors import PRIOR_BANK

BANK = dict(PRIOR_BANK)


async def bank_lift(teacher, prefix, y_c, z_a):
    keys = list(BANK)
    zs = [z_a] + [BANK[k] for k in keys]
    scores = await asyncio.gather(*[
        teacher.score_action(prefix, z, y_c) for z in zs
    ])
    lp_z = scores[0]["lp_per_byte"]
    lp_priors = [s["lp_per_byte"] for s in scores[1:]]
    return lp_z - max(lp_priors), lp_z - scores[1]["lp_per_byte"]  # bank, empty


async def run(args):
    prefixes = {}
    with open(args.turns) as f:
        for line in f:
            r = json.loads(line)
            prefixes[f"{r['traj_id']}:{r['turn_idx']}"] = r["prefix"]

    refs = {}
    with open(args.ref_cache) as f:
        for line in f:
            r = json.loads(line)
            refs[r["turn_id"]] = r["ref"]

    by_turn = defaultdict(list)
    with open(args.src) as f:
        for line in f:
            r = json.loads(line)
            if r.get("valid") and "pairs" in r:
                by_turn[r["turn_id"]].append(r)

    turn_ids = sorted(by_turn.keys())[: args.n_turns]
    rc = RunCfg()
    if args.concurrency:
        rc.max_concurrency = args.concurrency
    sem = asyncio.Semaphore(rc.max_concurrency)
    turn_sem = asyncio.Semaphore(max(4, min(args.turn_concurrency, 32)))
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for line in open(outp):
            r = json.loads(line)
            done.add((r["turn_id"], r["miner"]))

    lock = asyncio.Lock()
    out_f = open(outp, "a")
    finished = 0

    async with httpx.AsyncClient() as http:
        teacher = VllmModel(TEACHER, http, sem)

        async def one_turn(ti: int, tid: str) -> None:
            nonlocal finished
            prefix = prefixes.get(tid)
            ref = refs.get(tid)
            if not prefix or not ref:
                return
            async with turn_sem:
                for row in by_turn[tid]:
                    miner = row["miner"]
                    if (tid, miner) in done:
                        continue
                    m = min(len(row["pairs"]), len(ref))
                    lifts = await asyncio.gather(*[
                        bank_lift(teacher, prefix, ref[i]["y"],
                                  row["pairs"][i].get("z_a", ""))
                        for i in range(m)
                    ])
                    out = {
                        "turn_id": tid,
                        "miner": miner,
                        "L2_bank": sum(x[0] for x in lifts) / m,
                        "L2_empty": sum(x[1] for x in lifts) / m,
                        "n": m,
                    }
                    async with lock:
                        out_f.write(json.dumps(out) + "\n")
                        out_f.flush()
                        done.add((tid, miner))
            async with lock:
                finished += 1
                print(f"[{finished}/{len(turn_ids)}] {tid}", flush=True)

        await asyncio.gather(*[
            one_turn(ti, tid) for ti, tid in enumerate(turn_ids)
        ])
    out_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    ap.add_argument("--ref-cache", default="/root/results/ref_minicoder.jsonl")
    ap.add_argument("--n-turns", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=48,
                    help="max in-flight HTTP score_action calls")
    ap.add_argument("--turn-concurrency", type=int, default=12,
                    help="max turns scored in parallel")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
