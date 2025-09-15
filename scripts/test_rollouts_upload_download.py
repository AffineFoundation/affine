#!/usr/bin/env python3
import os
import time
import json
import uuid
import math
import asyncio
import logging
from typing import List, Tuple, Optional

from dotenv import load_dotenv

load_dotenv(".env.local")

import affine as af

# Configure logging similar to other scripts
logger = logging.getLogger("hippius_rollout_test")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(_handler)

# Small helper to build fake results
async def _make_fake_results(*, env_name: str, miner_hk: str, miner_uid: int, model: str, count: int) -> List[af.Result]:
    from affine.envs import ENVS as _ENVS
    env_cls = _ENVS[env_name]
    env = env_cls()
    results: List[af.Result] = []
    for i in range(count):
        prompt = f"Fake prompt {i} for {env_name}"
        chal = af.Challenge(env=env, prompt=prompt, extra={"fake": True, "i": i})
        # Simulate response/latency
        latency = 0.01 + (i % 5) * 0.005
        resp = af.Response(response="ok", latency_seconds=latency, attempts=1, model=model, error=None, success=True)
        # Score pattern in [0,1]
        score = (i % 10) / 10.0
        ev = af.Evaluation(env=env, score=score, extra={"source": "fake"})
        miner = af.Miner(uid=miner_uid, hotkey=miner_hk, model=model, revision="test")
        results.append(af.Result(miner=miner, challenge=chal, response=resp, evaluation=ev))
    return results

async def _upload_rollouts(wallet, results: List[af.Result]) -> Tuple[int, float]:
    t0 = time.perf_counter()
    await af.sink(wallet=wallet, results=results)
    dt = time.perf_counter() - t0
    return len(results), dt

async def _download_and_count(env_name: str, env_version: Optional[str], miner_hk: str) -> int:
    # Use reader aggregation pipeline to count samples for miner across shards
    pairs = [(miner_hk, None)]
    agg = await af.aggregate_scores_by_env(env_name=env_name, pairs=pairs, env_version=env_version)
    vals = agg.get(miner_hk) or {}
    return int(vals.get("n_total", 0))

async def _list_known_shards(env_name: str, env_version: Optional[str], miner_hk: str) -> List[str]:
    # Internals for metrics: list known shard keys
    from affine.database import get_client_ctx, _stream_root, _load_catalog_and_active, _shard_key, FOLDER
    root = _stream_root(miner_hk, env_name, env_version)
    async with get_client_ctx() as c:
        known, _active = await _load_catalog_and_active(c, root=root)
    return [_shard_key(root, seq) for seq in known]

async def main():
    # Inputs via env (dotenv) with defaults
    env_name = os.getenv("AFFINE_TEST_ENV", "SAT")
    miner_uid = int(os.getenv("AFFINE_TEST_MINER_UID", "12345"))
    model = os.getenv("AFFINE_TEST_MODEL", "test/model")
    count = int(os.getenv("AFFINE_TEST_COUNT", "50"))

    # Load existing wallet: name=default, hotkey=default
    import bittensor as bt
    wallet = bt.wallet(name="default", hotkey="default")
    miner_hk_env = (os.getenv("AFFINE_TEST_MINER_HK", "") or "").strip()
    miner_hk = miner_hk_env or wallet.hotkey.ss58_address

    logger.info(f"start rollout test env={env_name} miner={miner_hk} n={count}")

    # Generate fake results
    results = await _make_fake_results(env_name=env_name, miner_hk=miner_hk, miner_uid=miner_uid, model=model, count=count)

    # Upload
    n_up, dt_up = await _upload_rollouts(wallet, results)
    rps = n_up / max(dt_up, 1e-9)
    logger.info(f"uploaded {n_up} results in {dt_up:.3f}s ({rps:.1f} rps)")

    # Wait a moment for eventual consistency in listing
    await asyncio.sleep(1.0)

    # Determine env_version from results
    env_version = None
    if results:
        env_version = getattr(results[0].challenge.env, "__version__", None)

    # Download/aggregate
    t0 = time.perf_counter()
    n_total = await _download_and_count(env_name, env_version, miner_hk)
    dt_down = time.perf_counter() - t0
    logger.info(f"aggregated count for miner={miner_hk} on env={env_name}: {n_total} (took {dt_down:.3f}s)")

    # Metrics: shard sizes and keys involved
    try:
        keys = await _list_known_shards(env_name, env_version, miner_hk)
        logger.info(f"known shards for stream miner={miner_hk} env={env_name}: {len(keys)} keys")
        for k in keys[-5:]:  # show a few
            logger.info(f"shard: {k}")
    except Exception as e:
        logger.warning(f"failed to list shards: {e}")

    # Validate
    if n_total < n_up:
        raise SystemExit(f"Download/aggregation returned {n_total}; expected >= {n_up}")

    logger.info("rollout upload/download test OK")
    # Print compact JSON metrics for external parsing
    metrics = {
        "uploaded": n_up,
        "upload_seconds": dt_up,
        "upload_rps": rps,
        "aggregated_total": n_total,
        "download_seconds": dt_down,
    }
    print(json.dumps({"metrics": metrics}, separators=(",", ":")))

if __name__ == "__main__":
    asyncio.run(main())
