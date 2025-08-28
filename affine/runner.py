#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import time, random, asyncio, traceback, contextlib
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import bittensor as bt
import affine as af

# --------------------------------------------------------------------------- #
#                               Runner                                        #
# --------------------------------------------------------------------------- #
@af.cli.command("runner")
def runner():
    coldkey = af.get_conf("BT_WALLET_COLD", "default")
    hotkey  = af.get_conf("BT_WALLET_HOT",  "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)

    async def _run():
        
        envs = { env.__name__: env() for env in af.ENVS.values() }

        # State
        EPS = 1e-3
        MINERS: Dict[int, any] = None
        MINER_BY_PAIR: Dict[Tuple[str, str], any] = {}
        BACKOFF = defaultdict(float)

        async def refresh_miners():
            nonlocal MINERS, MINER_BY_PAIR
            MINERS = await af.get_miners()
            # build (hotkey, revision) → miner index
            MINER_BY_PAIR = { (m.hotkey, m.revision): m for m in MINERS.values() }

        COUNTS_PER_ENV: Dict[str, Dict[Tuple[str, str], int]] = None
        async def refresh_counts():
            nonlocal COUNTS_PER_ENV
            PAIRS = [ (m.hotkey, m.revision) for m in MINERS.values() ]
            COUNTS_PER_ENV = await af.get_env_counts(pairs=PAIRS)

        SELECT_LOG_TEMPLATE = (
            "[SELECT] "
            "U{uid:>3d} │ "
            "{model:<50s} │ "
            "{env:<3} │ "
            "CNT {count:>4d} │ "
            "WGT {weight:>12.4f} │ "
            "BKO {backoff:>6.1f}"
        )

        async def next():
            # Return if we still dont have any miners.
            if len(MINERS.values()) == 0 or len(COUNTS_PER_ENV.values()) == 0: return None
            # Get all hotkey, revision pairs to select from.
            pairs = [ (m.hotkey, m.revision) for m in MINERS.values() ]
            # Get a weight per env.
            weights_per_env = {env_name: 1/sum(env_counts.values()) for env_name, env_counts in COUNTS_PER_ENV.items()}
            # Select the env with the least number of samples.
            worst_env = random.choices( list(weights_per_env.keys()), weights = list(weights_per_env.values()))[0]
            # Get all counts for the worst env.
            env_counts = COUNTS_PER_ENV[worst_env]
            # Get the average env count.
            mean_env_count = sum([ c for c in env_counts.values() ])/(len(pairs) + EPS)
            # Weight to be selected is mean/(count + backoff)
            weights_hotkey_env = { p: mean_env_count/(env_counts.get(p, 0) + BACKOFF[p] + EPS) for p in pairs }
            # Pick the miner with weights.
            worst_miner = random.choices( list(weights_hotkey_env.keys()), weights = list(weights_hotkey_env.values()))[0]
            # return the env and the miner
            miner =  MINER_BY_PAIR[worst_miner]
            af.logger.debug(
                SELECT_LOG_TEMPLATE.format(
                    uid=miner.uid,
                    model=(miner.model or "")[:50],
                    env=worst_env,
                    count=COUNTS_PER_ENV[worst_env].get(worst_miner, 0),
                    weight=weights_hotkey_env[worst_miner],
                    backoff=BACKOFF[worst_miner],
                )
            )
            return worst_env, miner
                    
        while True:
            try:
                await refresh_miners()
                await refresh_counts()
                af.HEARTBEAT = time.monotonic()
                sink_semaphore = asyncio.Semaphore(3)
                inflight_semaphore = asyncio.Semaphore(30)
                async def one():
                    async with inflight_semaphore:
                        sel = await next()
                        if sel is None:return
                        env_name, miner = sel
                        pair = (miner.hotkey, miner.revision)                     # <<< added
                        chal = await envs[env_name].generate()
                        results = await af.run(chal, miner, timeout=180)
                        nobackoff = results[0].response.success and results[0].response != None and results[0].response != ""
                        if nobackoff: BACKOFF[pair] = max(0.0, BACKOFF[pair] - 10)
                        else: BACKOFF[pair] += 10
                        COUNTS_PER_ENV[env_name][pair] = COUNTS_PER_ENV[env_name].get(pair, 0) + 1
                        async with sink_semaphore:
                            await af.sink(wallet=wallet, results=results)
                await asyncio.gather(*[one() for _ in range(60)])
                    
            except Exception as e:
                af.logger.warning(f'Exception:{e}')
                af.logger.warning(traceback.format_exc())

    async def main():
        await asyncio.gather(_run(), af.watchdog(timeout=600))

    asyncio.run(main())
