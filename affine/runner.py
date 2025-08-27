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
        # ── config ───────────────────────────────────────────────────────────
        QUOTA_PER_ENV  = int(af.get_conf("AFFINE_ENV_QUOTA",     "1000"))  # target per env
        MAX_USES       = int(af.get_conf("AFFINE_CHAL_MAX_USES", "30"))    # #times to reuse a chal
        REFRESH_S      = int(af.get_conf("AFFINE_REFRESH_S",     "60"))    # metagraph/miners refresh cadence (s)
        SINK_BATCH     = int(af.get_conf("AFFINE_SINK_BATCH",    "50"))    # flush threshold
        SINK_MAX_WAIT  = int(af.get_conf("AFFINE_SINK_MAX_WAIT", str(60*5))) # max seconds to hold partial batch
        BACKOFF0       = float(af.get_conf("AFFINE_BACKOFF0",    "5"))     # base backoff (s)
        BACKOFF_CAP    = float(af.get_conf("AFFINE_BACKOFF_CAP", "300"))   # cap (s)
        DEAD_STREAK_N  = int(af.get_conf("AFFINE_DEAD_STREAK_N", "5"))     # consecutive fails → dead cooldown
        DEAD_COOLDOWN  = float(af.get_conf("AFFINE_DEAD_COOL_S", str(15*60)))  # s
        COUNT_CONCUR   = int(af.get_conf("AFFINE_COUNT_CONCUR",  "8"))     # concurrency for baseline counting
        RUN_TIMEOUT    = int(af.get_conf("AFFINE_RUN_TIMEOUT",   "180"))   # per-query timeout (s)

        # ── state: metagraph/miners/envs ─────────────────────────────────────
        subtensor = None
        env_objs: List[Any] = [cls() for cls in af.ENVS.values()]
        env_names: List[str] = [str(e) for e in env_objs]
        env_by_name: Dict[str, Any] = {str(e): e for e in env_objs}

        miners_map: Dict[int, Any] = {}
        last_sync: float = 0.0

        # chal reuse
        chal_cache: Dict[Any, Tuple[Any, int]] = {}  # env_obj -> (chal, uses)
        i_env_idx = 0

        # backoff & cooldown
        delay = defaultdict(lambda: BACKOFF0)    # uid -> current delay
        cooldown_until = defaultdict(float)      # uid -> time until allowed again
        fail_streak = defaultdict(int)           # uid -> consecutive fail streak

        # quotas
        baseline_per_env = {name: 0 for name in env_names}   # already in storage
        session_success  = {name: 0 for name in env_names}   # successes this run
        reserved         = {name: 0 for name in env_names}   # optimistic reservations

        # results pipeline
        sink_q: asyncio.Queue = asyncio.Queue()

        # monitoring
        total_requests = 0
        requests_since_log = 0
        last_status_log = 0.0

        # ── helpers ──────────────────────────────────────────────────────────
        def ok_any(res_list) -> bool:
            """Any successful result?"""
            try:
                for r in (res_list or []):
                    if getattr(r, "response", None) and getattr(r.response, "success", False):
                        return True
            except Exception:
                pass
            return False

        async def ensure_subtensor():
            nonlocal subtensor
            if subtensor is None:
                subtensor = await af.get_subtensor()
            return subtensor

        async def refresh_miners(now: float):
            nonlocal last_sync, miners_map
            if (now - last_sync) >= REFRESH_S or last_sync == 0:
                st = await ensure_subtensor()
                meta = await st.metagraph(af.NETUID)
                miners_map = await af.get_miners(meta=meta)
                last_sync = now
                af.logger.debug(f"refresh: miners={len(miners_map)}")

        async def next_chal_for(env_obj):
            """Reuse a challenge per env up to MAX_USES."""
            chal, uses = chal_cache.get(env_obj, (None, 0))
            if chal is None or uses >= MAX_USES:
                chal, uses = await env_obj.generate(), 0
            chal_cache[env_obj] = (chal, uses + 1)
            return chal

        def env_deficit(name: str) -> int:
            return max(0, QUOTA_PER_ENV - baseline_per_env[name] - session_success[name])

        def all_done() -> bool:
            return all(env_deficit(n) <= 0 for n in env_names)

        def choose_env_weighted() -> str | None:
            """Pick env ~ deficit weight."""
            items = [(n, env_deficit(n)) for n in env_names]
            items = [(n, d) if (d - reserved[n]) > 0 else (n, 0) for (n,d) in items]
            items = [(n, d) for (n,d) in items if d > 0]
            if not items:
                return None
            total = sum(d for _, d in items)
            x = random.uniform(0, total)
            s = 0.0
            for n, d in items:
                s += d
                if x <= s:
                    return n
            return items[-1][0]

        async def compute_baseline_counts():
            """Sum af.count over all miners for each env_name, with tight concurrency and 'too many clients' retries."""
            if not miners_map:
                return

            sem = asyncio.Semaphore(COUNT_CONCUR)

            async def _one(mi, env_name: str):
                async with sem:
                    # retry a few times if the DB says "too many clients"
                    backoff = 0.2
                    for attempt in range(6):  # ~ (0.2 + 0.4 + 0.8 + 1.6 + 3.2 + 6.4) ≈ 12.6s worst-case
                        try:
                            return env_name, await af.count(env_name=env_name, hotkey=mi.hotkey, revision=mi.revision)
                        except Exception as e:
                            msg = str(e).lower()
                            # only back off on overload-ish errors; otherwise bail
                            if "too many clients" in msg or "connection" in msg or "timeout" in msg:
                                jitter = random.uniform(0, backoff * 0.25)
                                af.logger.debug(f"count retry ({attempt+1}) for {env_name} uid={getattr(mi,'uid',None)}: {e} (sleep {backoff+jitter:.2f}s)")
                                await asyncio.sleep(backoff + jitter)
                                backoff = min(backoff * 2, 2.0)  # cap per-attempt sleep to keep progress moving
                                continue
                            # non-overload error → treat as 0 without retry storm
                            af.logger.debug(f"count error {env_name} uid={getattr(mi,'uid',None)}: {e}")
                            return env_name, 0
                    # after retries, give up for this (env, miner) pair
                    return env_name, 0

            tasks = []
            for env_name in env_names:
                for mi in miners_map.values():
                    tasks.append(asyncio.create_task(_one(mi, env_name)))

            agg = {n: 0 for n in env_names}
            for t in asyncio.as_completed(tasks):
                env_name, c = await t
                agg[env_name] += int(c or 0)

            for n in env_names:
                baseline_per_env[n] = agg[n]

            s = ", ".join(f"{n}:{baseline_per_env[n]}/{QUOTA_PER_ENV}" for n in env_names)
            af.logger.info(f"[BASELINE] {s}")

        async def sink_worker():
            """Batch + flush to storage."""
            batch = []
            first_put = None
            try:
                while True:
                    if first_put is None:
                        item = await sink_q.get()
                        first_put = time.monotonic()
                        batch.append(item)
                        # opportunistically drain
                        while len(batch) < SINK_BATCH:
                            try:
                                batch.append(sink_q.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                    else:
                        remaining = SINK_MAX_WAIT - (time.monotonic() - first_put)
                        timeout = remaining if remaining > 0.05 else 0.05
                        try:
                            item = await asyncio.wait_for(sink_q.get(), timeout=timeout)
                            batch.append(item)
                            while len(batch) < SINK_BATCH:
                                try:
                                    batch.append(sink_q.get_nowait())
                                except asyncio.QueueEmpty:
                                    break
                        except asyncio.TimeoutError:
                            pass

                    # debug pacing (optional, short sleep reduces log spam)
                    await asyncio.sleep(0.01)

                    elapsed = (time.monotonic() - first_put) if first_put is not None else 0.0
                    if len(batch) >= SINK_BATCH or (batch and elapsed >= SINK_MAX_WAIT):
                        st = await ensure_subtensor()
                        blk = await st.get_current_block()
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
                        batch.clear()
                        first_put = None
            except asyncio.CancelledError:
                # drain & final flush
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
                raise

        async def schedule(uid: int, miner, inflight: Dict[int, Tuple[asyncio.Task, str]], now: float):
            """Schedule one task for this miner if allowed; pick env by quota."""
            nonlocal total_requests, requests_since_log
            if uid in inflight: return
            if now < cooldown_until[uid]: return

            env_name = choose_env_weighted()
            if env_name is None:
                return  # nothing left to do

            env_obj = env_by_name[env_name]
            chal = await next_chal_for(env_obj)

            # reserve 1 slot for this env to avoid overshoot
            reserved[env_name] += 1

            t = asyncio.create_task(af.run(chal, miner, timeout=RUN_TIMEOUT))
            inflight[uid] = (t, env_name)
            total_requests += 1
            requests_since_log += 1

        async def main_loop():
            nonlocal last_status_log, requests_since_log

            # initialize miners + baseline
            now = time.monotonic()
            await refresh_miners(now)
            await compute_baseline_counts()

            inflight: Dict[int, Tuple[asyncio.Task, str]] = {}  # uid -> (task, env_name)
            sink_task = asyncio.create_task(sink_worker())

            try:
                while True:
                    now = time.monotonic()

                    # quit condition: all env quotas satisfied (incl baseline)
                    if all_done():
                        af.logger.info("[DONE] all env quotas satisfied; shutting down loop.")
                        break

                    # heartbeat & miners refresh
                    _ = await ensure_subtensor()
                    await refresh_miners(now)

                    # periodic status log
                    if now - last_status_log >= 30:
                        elapsed = (now - last_status_log) if last_status_log > 0 else 30
                        rps = requests_since_log / elapsed
                        cooldown_count = sum(1 for uid in miners_map.keys() if now < cooldown_until[uid])
                        queue_size = sink_q.qsize()
                        env_prog = " ".join(
                            f"{n}:{baseline_per_env[n]+session_success[n]}/{QUOTA_PER_ENV}"
                            for n in env_names
                        )
                        af.logger.info(
                            f"[STATUS] miners={len(miners_map)} inflight={len(inflight)} "
                            f"cooldown={cooldown_count} queue={queue_size} "
                            f"req/s={rps:.1f} total_req={total_requests} | {env_prog}"
                        )
                        last_status_log = now
                        requests_since_log = 0

                    # attempt to schedule one task per eligible miner
                    for uid, miner in list((int(m.uid), m) for m in miners_map.values()):
                        await schedule(uid, miner, inflight, now)

                    if not inflight:
                        # no tasks? maybe quotas are nearly met or miners cooling down
                        await asyncio.sleep(0.2)
                        continue

                    # wait for first completion
                    done, _ = await asyncio.wait([tk for tk, _ in inflight.values()],
                                                 return_when=asyncio.FIRST_COMPLETED)
                    now = time.monotonic()

                    # process completed tasks
                    for tk in done:
                        # find uid & env_name for this task
                        uid = None
                        env_name = None
                        for u, (t, en) in list(inflight.items()):
                            if t is tk:
                                uid, env_name = u, en
                                break
                        if uid is not None:
                            inflight.pop(uid, None)

                        # clear reservation for this env
                        if env_name:
                            reserved[env_name] = max(0, reserved[env_name] - 1)

                        miner = miners_map.get(uid)
                        try:
                            res_list = await tk  # list[Result]
                        except Exception as e:
                            af.logger.debug(f"miner {uid} task error: {e}")
                            res_list = []

                        # evaluate success
                        successes = 0
                        try:
                            for r in (res_list or []):
                                if getattr(r, "response", None) and getattr(r.response, "success", False):
                                    successes += 1
                        except Exception:
                            pass

                        if successes > 0:
                            # success path: reset backoff, account progress, enqueue results
                            delay[uid] = BACKOFF0
                            fail_streak[uid] = 0
                            cooldown_until[uid] = now  # allow immediate reschedule
                            if env_name:
                                session_success[env_name] += successes
                            try:
                                sink_q.put_nowait(res_list)  # sink will flatten
                            except asyncio.QueueFull:
                                # very unlikely with unbounded Queue, but just in case
                                await sink_q.put(res_list)
                            af.logger.debug(f"miner {uid} OK; +{successes} on {env_name}; queue={sink_q.qsize()}")
                        else:
                            # failure path: exponential backoff + jitter + dead cooldown on streak
                            fail_streak[uid] += 1
                            base = min(delay[uid] * 2, BACKOFF_CAP)
                            jitter = random.uniform(0, base * 0.2)
                            delay[uid] = base
                            cd = base + jitter
                            if fail_streak[uid] >= DEAD_STREAK_N:
                                cd = max(cd, DEAD_COOLDOWN)
                            cooldown_until[uid] = now + cd
                            af.logger.debug(f"miner {uid} FAIL; streak={fail_streak[uid]} cooldown={cd:.1f}s")

                        # opportunistic reschedule
                        if miner:
                            await schedule(uid, miner, inflight, now)

            except asyncio.CancelledError:
                pass
            finally:
                # cancel sink worker; ensure final flush
                sink_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sink_task

        await main_loop()

    async def main():
        await asyncio.gather(_run(), af.watchdog(timeout=600))

    asyncio.run(main())
