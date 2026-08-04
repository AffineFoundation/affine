"""RT-2: thought-stuffing attack. Does z_A := (text of y_A) improve the score?

For each turn we sample the honest miner once, then score the SAME actions under
three thought channels:
  honest   z = the miner's real reasoning
  stuffer  z = the action text itself ("I will run <command>")
  empty    z = ""
Attack succeeds where stuffer S > honest S. Run on the pod:
  python -m harness.rt2_stuffer --miner qwen3-8b=Qwen/Qwen3-8B:8010 \
      --n-turns 50 --out /root/results/rt2_stuffer.jsonl
"""

import argparse
import asyncio
import json
import pathlib

import httpx

from .client import VllmModel
from .config import RunCfg, TEACHER, ModelCfg
from .runner import load_turns, turn_id
from .terms import miner_terms, teacher_reference


def stuff(y: str) -> str:
    cmd = y.removeprefix("```bash\n").removesuffix("\n```").strip()
    return f"I will run: {cmd}"


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

            base = await asyncio.gather(*[
                miner.sample(prefix, rc.temperature,
                             rc.max_thought_tokens + rc.max_action_tokens)
                for _ in range(rc.n_miner_samples)
            ])
            base = [(z, y) for z, y in base if y]
            if not base:
                continue
            variants = {
                "honest": base,
                "stuffer": [(stuff(y), y) for _, y in base],
                "empty": [("", y) for _, y in base],
            }
            row = {"turn_id": tid, "causality": causality}
            for vname, rolls in variants.items():
                t = await miner_terms(teacher, miner, prefix, ref,
                                      rc.n_miner_samples, rc.temperature,
                                      rc.max_thought_tokens, rc.max_action_tokens,
                                      rollouts=rolls)
                row[vname] = t
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            print(f"[{ti + 1}/{len(turns)}] {tid}", flush=True)
        out_f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="/root/data/turns_minicoder.jsonl")
    ap.add_argument("--miner", required=True, help="name=repo:port")
    ap.add_argument("--n-turns", type=int, default=50)
    ap.add_argument("--ref-cache", default="/root/results/ref_minicoder.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
