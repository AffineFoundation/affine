"""End-to-end smoke test of the term estimator: teacher (:8000) vs Qwen3-8B (:8010).

Run on the pod: cd /root && python -m harness.smoke_test
"""

import asyncio
import json
import time

import httpx

from .client import VllmModel
from .config import TEACHER, ModelCfg, RunCfg
from .terms import miner_terms, teacher_reference

SMOKE_MINER = ModelCfg(name="qwen3-8b", repo="Qwen/Qwen3-8B", port=8010,
                       family="qwen", gpus="7", tp=1)


async def main():
    rc = RunCfg()
    sem = asyncio.Semaphore(rc.max_concurrency)
    turns = []
    with open("/root/data/turns_minicoder.jsonl") as f:
        for line in f:
            turns.append(json.loads(line))
            if len(turns) >= 3:
                break

    async with httpx.AsyncClient() as http:
        teacher = VllmModel(TEACHER, http, sem)
        miner = VllmModel(SMOKE_MINER, http, sem)
        for rec in turns:
            t0 = time.time()
            ref = await teacher_reference(
                teacher, rec["prefix"], 3, rc.temperature,
                rc.max_thought_tokens, rc.max_action_tokens)
            print(f"teacher rollouts: {len(ref)} in {time.time()-t0:.0f}s")
            if ref:
                r0 = ref[0]
                print("  z sample:", r0["z"][:120].replace("\n", " "))
                print("  y sample:", r0["y"][:120].replace("\n", " "))
                print(f"  lp_own={r0['lp_own']:.4f} lp_empty={r0['lp_empty']:.4f}")
            mz, my = await miner.sample(rec["prefix"], rc.temperature,
                                        rc.max_thought_tokens + rc.max_action_tokens)
            print("  miner z:", mz[:120].replace("\n", " "))
            print("  miner y:", my[:120].replace("\n", " "))
            t1 = time.time()
            terms = await miner_terms(
                teacher, miner, rec["prefix"], ref, 3, rc.temperature,
                rc.max_thought_tokens, rc.max_action_tokens)
            print(f"miner terms in {time.time()-t1:.0f}s: "
                  + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                for k, v in terms.items()}))


if __name__ == "__main__":
    asyncio.run(main())
