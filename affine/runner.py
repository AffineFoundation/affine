#!/usr/bin/env python3
from __future__ import annotations
import asyncio, random, time, traceback
from typing import Any, Dict, List
import statistics as stats
import affine as af

@af.cli.command("runner")
def runner():
    import bittensor as bt
    wallet  = bt.wallet(
        name=af.get_conf("BT_WALLET_COLD", "default"),
        hotkey=af.get_conf("BT_WALLET_HOT",  "default"),
    )

    # --- Tunables ------------------------------------------------------------
    CONCURRENCY         = int(af.get_conf("CONCURRENCY", 300))       # global cap
    ROUND_TIMEOUT_SEC   = float(af.get_conf("ROUND_TIMEOUT_SEC", 180))
    REFRESH_MINERS_SEC  = int(af.get_conf("REFRESH_MINERS_SEC", 600))
    SLEEP_EMPTY_SEC     = float(af.get_conf("SLEEP_EMPTY_SEC", 3))
    MAX_SAMPLE          = int(af.get_conf("MAX_SAMPLE", 0))         # 0 = all miners
    FAILURE_PENALTY     = float(af.get_conf("FAILURE_PENALTY", 0.0))
    USE_ZSCORE          = bool(int(af.get_conf("GRPO_USE_ZSCORE", 0)))
    LAUNCH_EVERY_SEC    = float(af.get_conf("LAUNCH_EVERY_SEC", 30)) # <-- new
    MAX_INFLIGHT_ROUNDS = int(af.get_conf("MAX_INFLIGHT_ROUNDS", 4)) # soft guard
    EPS                 = 1e-9

    # Build envs once; if your env set changes at runtime, add a refresher.
    def build_envs() -> Dict[str, Any]:
        return {E.__name__: E() for E in af.ENVS.values()}

    async def refresh_miners() -> Dict[int, Any]:
        return await af.get_miners()  # {uid -> MinerInfo}

    async def run_one(miner, chal):
        try:
            res = (await af.run(chal, miner, timeout=ROUND_TIMEOUT_SEC))[0]
            return miner, res
        except Exception:
            return miner, None

    def apply_grpo(results: List[Any]):
        succ = [r.evaluation.score for r in results if r and r.response.success]
        mean = (sum(succ)/len(succ)) if succ else 0.0
        if USE_ZSCORE and len(succ) > 1:
            std = stats.pstdev(succ) or EPS
            for r in results:
                if not r: 
                    continue
                if r.response.success:
                    r.evaluation.score = (r.evaluation.score - mean) / std
                else:
                    r.evaluation.score = (-mean - FAILURE_PENALTY) / std
        else:
            for r in results:
                if not r: 
                    continue
                if r.response.success:
                    r.evaluation.score = r.evaluation.score - mean
                else:
                    r.evaluation.score = -mean - FAILURE_PENALTY

    async def broadcast_round(env, miners: List[Any], req_sem: asyncio.Semaphore, round_id: int, env_name: str):
        """Fire one challenge to a cohort of miners; does not block subsequent rounds."""
        round_start_ts = time.monotonic()
        chal = await env.generate()

        cohort = miners[:] if MAX_SAMPLE <= 0 else random.sample(miners, min(MAX_SAMPLE, len(miners)))
        random.shuffle(cohort)

        async def bound(m):
            # Global concurrency guard across overlapping rounds
            async with req_sem:
                return await run_one(m, chal)

        # Kick off all miner requests for this round (bounded by req_sem)
        pairs = await asyncio.gather(*[asyncio.create_task(bound(m)) for m in cohort])

        # GRPO adjust on successful first responses
        first_results = [r for (_m, r) in pairs if r]
        grpo_avg = 0.0
        if first_results:
            apply_grpo(first_results)
            # Calculate average GRPO-adjusted score
            grpo_scores = [r.evaluation.score for r in first_results]
            grpo_avg = sum(grpo_scores) / len(grpo_scores)

        # Sink everything we got in one batch
        batch_results = [r for (_m, r) in pairs if r is not None]
        if batch_results:
            sink_start = time.monotonic()
            await af.sink(wallet=wallet, results=batch_results)
            sink_dur = time.monotonic() - sink_start
        else:
            sink_dur = 0.0

        # Round summary logging
        attempted = len(cohort)
        ok = sum(1 for (_m, r) in pairs if r and getattr(r.response, "success", False))
        fail = sum(1 for (_m, r) in pairs if r and not getattr(r.response, "success", False))
        err = sum(1 for (_m, r) in pairs if r is None)
        round_dur = time.monotonic() - round_start_ts
        af.logger.info(
            f"[round {round_id}] env={env_name} attempted={attempted} ok={ok} fail={fail} err={err} "
            f"dur={round_dur:.2f}s sink={len(batch_results)} in {sink_dur:.2f}s grpo_avg={grpo_avg:.3f}"
        )

    async def miner_refresher(state):
        """Periodically refresh miner list."""
        while True:
            try:
                state["miners"] = await refresh_miners()
            except Exception as e:
                af.logger.warning(f"[miner_refresher] {e}\n{traceback.format_exc()}")
            await asyncio.sleep(REFRESH_MINERS_SEC)

    async def launcher():
        """
        Launch a new round every LAUNCH_EVERY_SEC seconds, regardless of
        in-flight rounds. Uses a global semaphore to cap total requests.
        """
        envs = build_envs()
        env_names = list(envs.keys())
        idx = 0
        round_seq = 1

        # Shared state and controls
        state: Dict[str, Any] = {"miners": await refresh_miners()}
        req_sem = asyncio.Semaphore(CONCURRENCY)
        inflight_rounds: set[asyncio.Task] = set()

        # Start background miner refresher
        asyncio.create_task(miner_refresher(state))

        while True:
            try:
                miners_dict = state.get("miners") or {}
                miners_list = list(miners_dict.values())

                if not miners_list or not env_names:
                    await asyncio.sleep(SLEEP_EMPTY_SEC)
                    continue

                # Soft backpressure: avoid unbounded backlog of overlapping rounds
                # (Does NOT wait for previous round to finish; only limits # of concurrent rounds)
                if len(inflight_rounds) >= MAX_INFLIGHT_ROUNDS:
                    # Prune done tasks and try again quickly
                    done = {t for t in inflight_rounds if t.done()}
                    inflight_rounds -= done
                    await asyncio.sleep(0.05)
                    continue

                env_name = env_names[idx % len(env_names)]
                idx += 1

                af.HEARTBEAT = time.monotonic()

                # Launch round without awaiting its completion
                af.logger.info(
                    f"[round {round_seq}] launch env={env_name} miners={len(miners_list)} inflight={len(inflight_rounds)}/{MAX_INFLIGHT_ROUNDS}"
                )
                t = asyncio.create_task(broadcast_round(envs[env_name], miners_list, req_sem, round_seq, env_name))
                inflight_rounds.add(t)
                round_seq += 1

                # Cleanup finished tasks in the background
                def _done(_):
                    inflight_rounds.discard(t)
                t.add_done_callback(_done)

                # Tick at fixed cadence
                await asyncio.sleep(LAUNCH_EVERY_SEC)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                af.logger.warning(f"[launcher] Exception: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(0.5)

    async def main():
        await asyncio.gather(
            launcher(),
            af.watchdog(timeout=600),
        )

    asyncio.run(main())
