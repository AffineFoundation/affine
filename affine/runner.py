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

        # Tunables
        ALPHA = float(af.get_conf("AFFINE_DEFICIT_ALPHA", "2.0"))  # 0=uniform, 1=linear, 2=quadratic...
        TARGET_PER_ENV = int(af.get_conf("AFFINE_ENV_QUOTA", "1000"))
        EPS = float(af.get_conf("AFFINE_PROB_EPS", "1e-6"))        # small exploration mass
        BETA = float(af.get_conf("AFFINE_BACKOFF_BETA", "0.5"))    # <<< added (strength of backoff)
        DECAY = float(af.get_conf("AFFINE_BACKOFF_DECAY", "0.25")) # <<< added (amount to reduce on success)

        # State
        MINERS: Dict[int, any] = None
        MINER_BY_PAIR: Dict[Tuple[str, str], any] = {}
        BACKOFF = defaultdict(float)                               # <<< added: (hotkey,rev) -> backoff score

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

        async def next():
            """
            Probabilistically return (env_name, miner) where selection probability
            is proportional to ((TARGET_PER_ENV - count)^ALPHA + EPS) * backoff_factor.
            backoff_factor = 1 / (1 + BETA * BACKOFF[pair])
            """
            if not MINERS or not COUNTS_PER_ENV:
                return None
            choices = []
            weights = []
            for env_name, env_counts in COUNTS_PER_ENV.items():
                for pair, count in env_counts.items():
                    miner = MINER_BY_PAIR.get(pair)
                    if miner is None:
                        continue
                    deficit = TARGET_PER_ENV - int(count or 0)
                    base_w = (max(deficit, 0) ** ALPHA) + EPS
                    damp = 1.0 / (1.0 + (BETA * BACKOFF[pair]))
                    w = base_w * damp
                    if w <= 0.0:
                        continue
                    choices.append((env_name, miner))
                    weights.append(w)

            if not choices:return None
            pick = random.choices(choices, weights=weights, k=1)[0]
            return pick

        
        while True:
            try:
                await refresh_miners()
                await refresh_counts()
                sink_semaphore = asyncio.Semaphore(3)
                inflight_semaphore = asyncio.Semaphore(30)
                async def one():
                    async with inflight_semaphore:
                        sel = await next()
                        if sel is None:return
                        env_name, miner = sel
                        pair = (miner.hotkey, miner.revision)                     # <<< added
                        chal = await envs[env_name].generate()
                        results = await af.run(chal, miner)
                        nobackoff = results[0].response.success and results[0].response != None and results[0].response != ""
                        if nobackoff: BACKOFF[pair] = max(0.0, BACKOFF[pair] - DECAY)
                        else: BACKOFF[pair] += 1.0
                        COUNTS_PER_ENV[env_name][pair] = COUNTS_PER_ENV[env_name].get(pair, 0) + 1
                        async with sink_semaphore:
                            await af.sink(wallet=wallet, results=results)
                await asyncio.gather(*[one() for _ in range(60)])
                    
            except Exception as e:
                af.logger.warning(f'Exception:{e}')

    async def main():
        await asyncio.gather(_run(), af.watchdog(timeout=600))

    asyncio.run(main())
