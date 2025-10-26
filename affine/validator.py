import os
import time
import socket
import asyncio
import logging
import aiohttp
import bittensor as bt
from typing import List, Tuple
from urllib.parse import urlparse
from aiohttp import ClientConnectorError
from tabulate import tabulate
from affine.storage import prune, dataset
from affine.config import get_conf
from affine.setup import NETUID
from affine.utils.subtensor import get_subtensor
from affine.sampling import MinerSampler, SamplingOrchestrator, SamplingConfig
from affine.miners import miners
from affine.setup import ENVS, logger


async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 15,
    delay_s: float = 2.0,
    log_prefix: str = "",
    confirmation_blocks: int = 3,
) -> bool:
    for attempt in range(retries):
        try:
            st = await get_subtensor()
            logger.info(f"{log_prefix} set_weights attempt {attempt+1}/{retries}: netuid={netuid} uids={uids} weights={weights}")

            # Get block state before submission for debugging
            pre_block = await st.get_current_block()
            logger.info(f"{log_prefix} current block before submission: {pre_block}")

            # Submit weights using sync subtensor
            start = time.monotonic()
            sync_st = bt.subtensor(get_conf('SUBTENSOR_ENDPOINT', default='finney'))
            sync_st.set_weights(
                wallet=wallet, netuid=netuid, weights=weights, uids=uids,
                wait_for_inclusion=wait_for_inclusion,
            )
            submit_duration = (time.monotonic() - start) * 1000

            # Get reference block immediately after submission
            ref = await st.get_current_block()
            logger.info(f"{log_prefix} extrinsic submitted in {submit_duration:.1f}ms; ref_block={ref} (pre_block={pre_block})")

            # Wait for multiple blocks to ensure transaction inclusion
            for i in range(confirmation_blocks):
                await st.wait_for_block()
                current_block = await st.get_current_block()
                logger.info(f"{log_prefix} waited block {i+1}/{confirmation_blocks}, current_block={current_block}")

            # Verify weights have been updated
            meta = await st.metagraph(netuid)
            try:
                idx = meta.hotkeys.index(wallet.hotkey.ss58_address)
                lu = meta.last_update[idx]
                final_block = await st.get_current_block()
                logger.info(f"{log_prefix} last_update={lu}, ref_block={ref}, final_block={final_block}")

                # Check if last_update was updated within reasonable range
                if lu >= ref:
                    logger.info(f"{log_prefix} confirmation OK (last_update={lu} >= ref={ref})")
                    return True
                else:
                    logger.warning(f"{log_prefix} confirmation not yet included (last_update={lu} < ref={ref}), retrying …")
            except ValueError:
                logger.warning(f"{log_prefix} wallet hotkey not found in metagraph hotkeys; retrying …")
        except Exception as e:
            logger.warning(f"{log_prefix} set_weights attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}")

        await asyncio.sleep(delay_s)
    logger.error(f"{log_prefix} failed to confirm set_weights after {retries} attempts")
    return False


async def retry_set_weights( wallet: bt.Wallet, uids: List[int], weights: List[float], retry: int = 10 ):
    # Delegate to signer; fallback to shared helper only if signer is unreachable
    signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
    try:
        logger.info(f"Calling signer at {signer_url} for set_weights uids={uids}, weights={weights}")
        parsed = urlparse(signer_url)
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
            addrs = ",".join(sorted({i[4][0] for i in infos}))
            logger.info(f"Signer DNS: host={parsed.hostname} -> {addrs}")
        except Exception as e:
            logger.warning(f"DNS resolve failed for {parsed.hostname}: {e}")
        timeout = aiohttp.ClientTimeout(connect=2, total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.monotonic()
            resp = await session.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": NETUID,
                    "weights": weights,
                    "uids": uids,
                    "wait_for_inclusion": False,
                },
            )
            dur_ms = (time.monotonic() - start) * 1000
            logger.info(f"Signer HTTP response status={resp.status} in {dur_ms:.1f}ms")
            # Try to parse JSON, otherwise log text (trimmed)
            try:
                data = await resp.json()
            except Exception:
                txt = await resp.text()
                data = {"raw": (txt[:500] + ('…' if len(txt) > 500 else ''))}
            logger.info(f"Signer response body={data}")
            if resp.status == 200 and data.get("success"):
                return
            # Do not fallback if signer exists but reports failure
            logger.warning(f"Signer responded error: status={resp.status} body={data}")
            return
    except ClientConnectorError as e:
        logger.info(f"Signer not reachable ({type(e).__name__}: {e}); falling back to local set_weights once")
        ok = await _set_weights_with_confirmation(
            wallet, NETUID, uids, weights, False,
            retries=int(os.getenv("SIGNER_RETRIES", "10")),
            delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
            confirmation_blocks=int(os.getenv("CONFIRMATION_BLOCKS", "3")),
            log_prefix="[validator-fallback]",
        )
        if not ok:
            logger.error("Local set_weights confirmation failed")
        return
    except asyncio.TimeoutError as e:
        logger.warning(f"Signer call timed out: {e}. Not falling back to local because validator has no wallet.")
        return

async def get_weights(tail: int = SamplingConfig.TAIL, burn: float = 0.0):
    burn = max(0.0, min(1.0, burn))
    if burn >= 1:
        logger.info(f"Burn all")
        return [0], [1.0]

    sampler = MinerSampler()
    orchestrator = SamplingOrchestrator(sampler)
    
    st = await get_subtensor()
    blk = await st.get_current_block()
    logger.info(f"Pruning {tail} blocks from {blk - tail} to {blk}")
    await prune(tail=tail)

    meta = await st.metagraph(NETUID)
    BASE_HK = meta.hotkeys[0]
    N_envs = len(ENVS)
    
    queryable_miners = await miners(meta=meta, netuid=NETUID, check_validity=True)
    queryable_hks = {m.hotkey for m in queryable_miners.values()}
    logger.info(f"Found {len(queryable_hks)} queryable miners (hot, valid chute, not gated)")

    results_list = []
    
    initial_first_block = {}
    try:
        commits = await st.get_all_revealed_commitments(NETUID)
        for uid, hk in enumerate(meta.hotkeys):
            if hk in commits:
                blk_commit, _ = commits[hk][-1]
                initial_first_block[hk] = 0 if uid == 0 else int(blk_commit)
    except Exception:
        pass

    logger.info(f"Loading data from {blk - tail} to {blk}")
    async for c in dataset(tail=tail):
        results_list.append(c)

    logger.info("Collected results.")

    if not results_list:
        logger.warning("No results collected; defaulting to uid 0")
        return [0], [1.0]
    
    cnt, succ, prev, v_id, first_block, stats = orchestrator.process_sample_data(
        results_list, meta.hotkeys, ENVS, BASE_HK
    )
    
    for hk, blk_val in initial_first_block.items():
        if hk not in first_block:
            first_block[hk] = blk_val
    
    acc = orchestrator.calculate_accuracies(cnt, succ, meta.hotkeys, ENVS)
    
    # Calculate confidence intervals for all miners using Beta distribution
    confidence_intervals = {}
    for hk in meta.hotkeys:
        confidence_intervals[hk] = {}
        for e in ENVS:
            if cnt[hk][e] > 0:
                lower, upper = sampler.challenge_algo.beta_confidence_interval(
                    succ[hk][e], cnt[hk][e]
                )
                confidence_intervals[hk][e] = (lower, upper)
            else:
                confidence_intervals[hk][e] = (0.0, 0.0)
    
    active_hks = list(prev.keys())
    logger.info("Computed accuracy.")

    eligible, required = sampler.calculate_eligibility(cnt, active_hks, queryable_hks, ENVS)
    logger.info(f"Eligible miners: {len(eligible)} (from {len(active_hks)} active, {len(queryable_hks)} queryable)")

    pool_for_dom = eligible if eligible else (queryable_hks & set(active_hks))

    score, layer_points, env_winners = sampler.calculate_combinatoric_scores(
        ENVS, pool_for_dom, stats, confidence_intervals
    )

    if not eligible:
        logger.warning(f"No eligible miners (queryable={len(queryable_hks)}); assigning weight 1.0 to uid 0.")
        
        hdr = (
            ["UID", "Model", "Rev"]
            + [f"{e}" for e in ENVS]
            + [f"L{s}" for s in range(1, N_envs + 1)]
            + ["Pts", "Elig", "FirstBlk", "Wgt"]
        )
        def row(hk: str):
            if hk not in prev:
                return None
            m = prev[hk].miner
            w = 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for e in ENVS:
                lower, upper = confidence_intervals[hk][e]
                base = f"{100 * acc[hk][e]:.2f}/[{100 * lower:.2f},{100 * upper:.2f}]/{cnt[hk][e]}"
                if hk == env_winners.get(e):
                    env_cols.append(f"*{base}*")
                else:
                    env_cols.append(base)
            return [
                m.uid, model_name, str(m.revision)[:5],
                *env_cols,
                *[f"{layer_points[hk][s]:.1f}" for s in range(1, N_envs + 1)],
                f"{score.get(hk, 0.0):.2f}",
                "Y" if hk in eligible else "N",
                f"{first_block.get(hk, 0)}",
                f"{w:.4f}",
            ]
        rows = sorted((r for r in (row(hk) for hk in active_hks) if r is not None), key=lambda r: (r[-4], r[0]), reverse=True)
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))
        return [0], [1.0]

    weight_by_hk, eligible = orchestrator.calculate_weights(eligible, score, burn, BASE_HK)

    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in ENVS]
        + [f"L{s}" for s in range(1, N_envs + 1)]
        + ["Pts", "Elig", "FirstBlk", "Wgt"]
    )
    def row(hk: str):
        if hk not in prev:
            return None
        m = prev[hk].miner
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for e in ENVS:
            lower, upper = confidence_intervals[hk][e]
            base = f"{100 * acc[hk][e]:.2f}/[{100 * lower:.2f},{100 * upper:.2f}]/{cnt[hk][e]}"
            if hk == env_winners.get(e):
                env_cols.append(f"*{base}*")
            else:
                env_cols.append(base)
        return [
            m.uid, model_name[:30], str(m.revision)[:5],
            *env_cols,
            *[f"{layer_points[hk][s]:.1f}" for s in range(1, N_envs + 1)],
            f"{score.get(hk, 0.0):.2f}",
            "Y" if hk in eligible else "N",
            f"{first_block.get(hk, 0)}",
            f"{w:.4f}",
        ]
    ranked_rows   = sorted((r for r in (row(hk) for hk in eligible) if r is not None), key=lambda r: float(r[-4]), reverse=True)
    unranked_rows = sorted((r for r in (row(hk) for hk in active_hks if hk not in eligible) if r is not None), key=lambda r: float(r[-4]), reverse=True)
    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"), flush=True)

    uids = [meta.hotkeys.index(hk) for hk in eligible]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights
