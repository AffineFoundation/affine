#!/usr/bin/env python3
import os
import time
import json
import asyncio
import logging
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv(".env.local")

import affine as af
import bittensor as bt

logger = logging.getLogger("runner_like_test")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(_handler)

async def make_fake_results(env_name: str, miner_hk: str, n: int) -> List[af.Result]:
    # Use real env class and simple constant score/success
    from affine.envs import ENVS as _ENVS
    env = _ENVS[env_name]()
    results: List[af.Result] = []
    for i in range(n):
        chal = af.Challenge(env=env, prompt=f"test #{i}")
        resp = af.Response(response="ok", latency_seconds=0.01, attempts=1, model="test/model", error=None, success=True)
        ev = af.Evaluation(env=env, score=1.0, extra={"source": "runner_like_test"})
        miner = af.Miner(uid=i, hotkey=miner_hk, model="test/model", revision="test")
        results.append(af.Result(miner=miner, challenge=chal, response=resp, evaluation=ev))
    return results

async def main():
    # Choose env and fetch a real hotkey from current metagraph so validator paths can see it
    env_name = os.getenv("AFFINE_TEST_ENV", "SAT")
    st = await af.get_subtensor()
    meta = await st.metagraph(af.NETUID)
    if not meta.hotkeys:
        raise SystemExit("No hotkeys in metagraph; cannot run test")
    miner_hk = meta.hotkeys[0]

    # Upload N results
    N = int(os.getenv("AFFINE_TEST_COUNT", "10"))
    wallet = bt.wallet(name=os.getenv("BT_WALLET_COLD", "default"), hotkey=os.getenv("BT_WALLET_HOT", "default"))
    results = await make_fake_results(env_name, miner_hk, N)
    t0 = time.perf_counter()
    await af.sink(wallet=wallet, results=results)
    dt_up = time.perf_counter() - t0
    logger.info(f"sank {N} results for hk={miner_hk} env={env_name} in {dt_up:.2f}s")

    # Verify aggregate sees them (may include previous counts)
    pairs = [(miner_hk, None)]
    t1 = time.perf_counter()
    agg = await af.aggregate_scores_by_env(env_name=env_name, pairs=pairs, env_version=getattr(af.ENVS[env_name], "__version__", None))
    dt_ag = time.perf_counter() - t1
    n_total = int((agg.get(miner_hk) or {}).get("n_total", 0))
    logger.info(f"aggregate for hk={miner_hk} env={env_name}: n_total={n_total} in {dt_ag:.2f}s")

    print(json.dumps({
        "env": env_name,
        "miner_hotkey": miner_hk,
        "uploaded": N,
        "upload_seconds": dt_up,
        "aggregated_total": n_total,
        "aggregate_seconds": dt_ag,
    }, separators=(",", ":")))

if __name__ == "__main__":
    asyncio.run(main())
