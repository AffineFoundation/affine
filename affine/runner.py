
#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import time
import random
import asyncio
import traceback
import contextlib
from .utils import *
import bittensor as bt
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable, AsyncIterator

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
        subtensor = None
        envs = [cls() for cls in af.ENVS.values()]

        # ── config ───────────────────────────────────────────────────────────
        MAX_USES       = 30
        REFRESH_S      = 600     # metagraph/miners refresh cadence (s)
        SINK_BATCH     = 300     # flush threshold
        SINK_MAX_WAIT  = 60*5      # max seconds to hold partial batch
        BACKOFF0       = 5
        BACKOFF_CAP    = 300

        # ── state ───────────────────────────────────────────────────────────
        chal_cache, i_env = {}, 0
        last_sync = 0.0
        delay = defaultdict(lambda: BACKOFF0)   # uid -> current delay
        cooldown_until = defaultdict(float)     # uid -> t when allowed again
        miners_map = {}

        # results pipeline
        sink_q: asyncio.Queue = asyncio.Queue()

        # monitoring state
        last_status_log = 0.0
        total_requests = 0
        requests_since_last_log = 0

        def ok(res_list):
            if not res_list: return False
            r = res_list[0]
            if not r.response.success: return False
            return True

        async def next_chal():
            nonlocal i_env
            e = envs[i_env]; i_env = (i_env + 1) % len(envs)
            chal, uses = chal_cache.get(e, (None, 0))
            if chal is None or uses >= MAX_USES:
                chal, uses = await e.generate(), 0
            chal_cache[e] = (chal, uses + 1)
            return chal

        async def schedule(miner, inflight, now):
            nonlocal total_requests, requests_since_last_log
            uid = int(miner.uid)
            if uid in inflight: return
            if now < cooldown_until[uid]: return
            chal = await next_chal()
            inflight[uid] = asyncio.create_task(af.run(chal, miner, timeout=180))
            total_requests += 1
            requests_since_last_log += 1

        async def ensure_subtensor():
            nonlocal subtensor
            if subtensor is None:
                subtensor = await af.get_subtensor()
            return subtensor

        async def refresh_miners(now):
            nonlocal last_sync, miners_map
            if (now - last_sync) >= REFRESH_S or last_sync == 0:
                st = await ensure_subtensor()
                meta = await st.metagraph(af.NETUID)
                miners_map = await af.get_miners(meta=meta)
                last_sync = now
                af.logger.debug(f"refresh: miners={len(miners_map)}")

        async def sink_worker():
            """Consumes results from sink_q and flushes in batches of SINK_BATCH or after SINK_MAX_WAIT."""
            nonlocal subtensor
            batch = []
            first_put_time = None
            while True:
                try:
                    # If we have started a batch, only wait up to the remaining hold time; otherwise wait for first item.
                    if first_put_time is None:
                        af.logger.debug(f"sink_worker: queue size={sink_q.qsize()}")
                        item = await sink_q.get()
                        first_put_time = time.monotonic()
                        batch.append(item)
                        # Opportunistically drain without blocking to build the batch quickly
                        while len(batch) < SINK_BATCH:
                            try:
                                more = sink_q.get_nowait()
                                batch.append(more)
                            except asyncio.QueueEmpty:
                                break
                    else:
                        remaining = SINK_MAX_WAIT - (time.monotonic() - first_put_time)
                        timeout = remaining if remaining > 0.05 else 0.05
                        try:
                            item = await asyncio.wait_for(sink_q.get(), timeout=timeout)
                            batch.append(item)
                            while len(batch) < SINK_BATCH:
                                try:
                                    more = sink_q.get_nowait()
                                    batch.append(more)
                                except asyncio.QueueEmpty:
                                    break
                        except asyncio.TimeoutError:
                            pass

                    elapsed = (time.monotonic() - first_put_time) if first_put_time is not None else 0.0
                    af.logger.debug(f"Until Sink: {len(batch)}/{SINK_BATCH} Time: {elapsed}/{SINK_MAX_WAIT}")
                    await asyncio.sleep(3)
                    if len(batch) >= SINK_BATCH or (batch and elapsed >= SINK_MAX_WAIT):
                        st = await ensure_subtensor()
                        blk = await st.get_current_block()
                        # Flatten: items may be single Result or list[Result]
                        flat = []
                        for it in batch:
                            if isinstance(it, list):
                                flat.extend(it)
                            else:
                                flat.append(it)
                        af.logger.debug(f"sink_worker: flushing {len(flat)} results")
                        try:
                            await af.sink(wallet=wallet, block=blk, results=flat)
                        except Exception:
                            traceback.print_exc()
                            # keep going; don't drop future batches
                        batch.clear()
                        first_put_time = None
                except asyncio.CancelledError:
                    # drain and final flush
                    flat = []
                    while not sink_q.empty():
                        it = sink_q.get_nowait()
                        if isinstance(it, list): flat.extend(it)
                        else: flat.append(it)
                    if flat:
                        try:
                            st = await ensure_subtensor()
                            blk = await st.get_current_block()
                            af.logger.debug(f"sink_worker: final flush {len(flat)}")
                            await af.sink(wallet=wallet, block=blk, results=flat)
                        except Exception:
                            traceback.print_exc()
                    break

        async def main_loop():
            nonlocal last_status_log, requests_since_last_log
            inflight = {}
            sink_task = asyncio.create_task(sink_worker())
            try:
                while True:
                    now = time.monotonic()
                    # heartbeat + ensure subtensor
                    _ = await ensure_subtensor()
                    # periodic refresh
                    await refresh_miners(now)
                    if not miners_map:
                        await asyncio.sleep(1)
                        continue

                    # periodic status logging
                    if now - last_status_log >= 30:
                        elapsed = now - last_status_log if last_status_log > 0 else 30
                        rps = requests_since_last_log / elapsed
                        cooldown_count = sum(1 for uid in miners_map.keys() if now < cooldown_until[uid])
                        queue_size = sink_q.qsize()
                        af.logger.info(f"[STATUS] miners={len(miners_map)} inflight={len(inflight)} cooldown={cooldown_count} queue={queue_size} req/s={rps:.1f} total_req={total_requests}")
                        last_status_log = now
                        requests_since_last_log = 0

                    # seed/respect cooldowns
                    for m in miners_map.values():
                        await schedule(m, inflight, now)

                    if not inflight:
                        await asyncio.sleep(0.2)
                        continue

                    done, _ = await asyncio.wait(inflight.values(), return_when=asyncio.FIRST_COMPLETED)
                    now = time.monotonic()
                    for t in done:
                        uid = next((u for u, tk in list(inflight.items()) if tk is t), None)
                        miner = miners_map.get(uid)
                        inflight.pop(uid, None)
                        try:
                            res_list = await t  # list[Result]; may be []
                        except Exception as e:
                            af.logger.debug(f"miner {uid} task error: {e}")
                            res_list = []

                        if ok(res_list):
                            # reset backoff, enqueue results (non-blocking)
                            delay[uid] = BACKOFF0
                            cooldown_until[uid] = now
                            # push entire list; sink worker will flatten
                            sink_q.put_nowait(res_list)
                            queue_size = sink_q.qsize()
                            af.logger.debug(f"miner {uid} OK; queued {len(res_list)}, queue_size={queue_size}")
                        else:
                            print ('not ok')
                            # exponential backoff + jitter
                            d = min(delay[uid] * 2, BACKOFF_CAP)
                            jitter = random.uniform(0, d * 0.2)
                            delay[uid] = d
                            cooldown_until[uid] = now + d + jitter
                            af.logger.debug(f"miner {uid} FAIL; cooldown {d:+.1f}s(+{jitter:.1f})")

                        # try to reschedule
                        if miner:
                            await schedule(miner, inflight, now)
            except asyncio.CancelledError:
                pass
            finally:
                # cancel sink worker and wait for final flush
                sink_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sink_task

        await main_loop()

    async def main():
        await asyncio.gather(_run(), af.watchdog(timeout=600))

    asyncio.run(main())