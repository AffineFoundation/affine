import os
import time
import socket
import asyncio
import logging
import aiohttp
import bittensor as bt
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass
from urllib.parse import urlparse
from aiohttp import ClientConnectorError
from tabulate import tabulate
from affine.storage import prune, dataset, save_summary
from affine.config import get_conf
from affine.setup import NETUID
from affine.utils.subtensor import get_subtensor
from affine.sampling import MinerSampler, SamplingOrchestrator, SamplingConfig
from affine.miners import miners
from affine.setup import ENVS, logger

# Table structure constants
BASE_COLUMNS = ["UID", "Model", "Rev"]
ENV_START_INDEX = len(BASE_COLUMNS)


@dataclass
class SummaryContext:
    """Context data for building validator summary."""
    block: int
    header: list
    rows: list
    eligible: Set[str]
    active_hotkeys: Set[str]
    queryable_hotkeys: Set[str]
    env_winners: Dict[str, str]
    accuracies: Dict[str, Dict[str, float]]
    counts: Dict[str, Dict[str, int]]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    scores: Dict[str, float]
    layer_points: Dict[str, Dict[int, float]]
    first_blocks: Dict[str, int]
    metagraph: Any
    previous_miners: Dict[str, Any]
    note: Optional[str] = None


def _create_miner_row(
    hotkey: str,
    prev: Dict,
    weight_by_hk: Dict[str, float],
    acc: Dict,
    cnt: Dict,
    confidence_intervals: Dict,
    env_winners: Dict,
    layer_points: Dict,
    score: Dict,
    eligible: Set[str],
    first_block: Dict,
    envs: list,
    n_envs: int,
    model_name_max_len: int = 50
) -> Optional[list]:
    """Create a single row for miner display table.

    Args:
        hotkey: Miner's hotkey
        prev: Previous miners data
        weight_by_hk: Weight mapping by hotkey
        acc: Accuracy data by hotkey and environment
        cnt: Count data by hotkey and environment
        confidence_intervals: Confidence interval data
        env_winners: Environment winners mapping
        layer_points: Layer points by hotkey
        score: Score mapping by hotkey
        eligible: Set of eligible hotkeys
        first_block: First block mapping by hotkey
        envs: List of environment names
        n_envs: Number of environments
        model_name_max_len: Maximum length for model name display

    Returns:
        List representing a table row, or None if hotkey not in prev
    """
    if hotkey not in prev:
        return None

    miner = prev[hotkey].miner
    weight = weight_by_hk.get(hotkey, 0.0)
    model_name = str(miner.model)[:model_name_max_len]

    env_cols = []
    for e in envs:
        lower, upper = confidence_intervals[hotkey][e]
        base = f"{100 * acc[hotkey][e]:.2f}/[{100 * lower:.2f},{100 * upper:.2f}]/{cnt[hotkey][e]}"
        if hotkey == env_winners.get(e):
            env_cols.append(f"*{base}*")
        else:
            env_cols.append(base)

    return [
        miner.uid,
        model_name,
        str(miner.revision)[:5],
        *env_cols,
        *[f"{layer_points[hotkey][s]:.1f}" for s in range(1, n_envs + 1)],
        f"{score.get(hotkey, 0.0):.2f}",
        "Y" if hotkey in eligible else "N",
        f"{first_block.get(hotkey, 0)}",
        f"{weight:.4f}",
    ]


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

def _build_summary_data(ctx: SummaryContext) -> dict:
    """Build flexible summary data structure.

    Args:
        ctx: SummaryContext containing all necessary data

    Returns:
        Dictionary with both legacy format (for printing) and structured format (for S3).
    """
    # Build structured miners data
    miners_data = {}
    for row in ctx.rows:
        if not row:
            continue
        uid = row[0]
        try:
            hotkey = ctx.metagraph.hotkeys[uid]
            if hotkey not in ctx.previous_miners:
                continue
            miner = ctx.previous_miners[hotkey].miner

            # Parse environment results from row
            env_results = {}
            for i, e in enumerate(ENVS):
                env_col_idx = ENV_START_INDEX + i
                env_results[e] = {
                    "accuracy": ctx.accuracies[hotkey][e],
                    "count": ctx.counts[hotkey][e],
                    "confidence_interval": {
                        "lower": ctx.confidence_intervals[hotkey][e][0],
                        "upper": ctx.confidence_intervals[hotkey][e][1]
                    },
                    "is_winner": (hotkey == ctx.env_winners.get(e))
                }

            miners_data[hotkey] = {
                "uid": uid,
                "hotkey": hotkey,
                "model": str(miner.model),
                "revision": str(miner.revision),
                "environments": env_results,
                "layer_points": {f"L{s}": ctx.layer_points[hotkey][s] for s in range(1, len(ENVS) + 1)},
                "total_score": ctx.scores.get(hotkey, 0.0),
                "eligible": hotkey in ctx.eligible,
                "first_block": ctx.first_blocks.get(hotkey, 0),
                "weight": float(row[-1]) if len(row) > 0 else 0.0
            }
        except (IndexError, KeyError, ValueError) as e:
            logger.debug(f"Skipping row for UID {uid} due to error: {e}")
            continue

    # Build summary structure
    summary = {
        "header": ctx.header,
        "rows": ctx.rows,  # Keep legacy format for compatibility
        "miners": miners_data,  # New structured format
        "stats": {
            "eligible_count": len(ctx.eligible),
            "active_count": len(ctx.active_hotkeys),
            "queryable_count": len(ctx.queryable_hotkeys),
            "total_miners": len(ctx.metagraph.hotkeys)
        },
        "env_winners": {e: ctx.env_winners.get(e) for e in ENVS},
        "environments": list(ENVS)
    }

    if ctx.note:
        summary["note"] = ctx.note

    return summary

async def get_weights(tail: int = SamplingConfig.TAIL, burn: float = 0.0, save_to_s3: bool = True):
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
    except Exception as e:
        logger.warning(f"Failed to get revealed commitments, using empty initial_first_block: {type(e).__name__}: {e}")

    logger.info(f"Loading data from {blk - tail} to {blk}")
    async for c in dataset(tail=tail, compact=True):
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

        # Use shared row creation function
        weight_by_hk = {}  # Empty weights for no eligible case
        rows = sorted(
            (r for r in (
                _create_miner_row(
                    hk, prev, weight_by_hk, acc, cnt, confidence_intervals,
                    env_winners, layer_points, score, eligible, first_block,
                    ENVS, N_envs, model_name_max_len=50
                ) for hk in active_hks
            ) if r is not None),
            key=lambda r: (r[-4], r[0]),
            reverse=True
        )
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

        # Save summary to S3 (no eligible miners case)
        ctx = SummaryContext(
            block=blk,
            header=hdr,
            rows=rows,
            eligible=set(),
            active_hotkeys=active_hks,
            queryable_hotkeys=queryable_hks,
            env_winners=env_winners,
            accuracies=acc,
            counts=cnt,
            confidence_intervals=confidence_intervals,
            scores=score,
            layer_points=layer_points,
            first_blocks=first_block,
            metagraph=meta,
            previous_miners=prev,
            note="No eligible miners, defaulting to uid 0"
        )
        summary_data = _build_summary_data(ctx)
        await save_summary(blk, summary_data)

        return [0], [1.0]

    weight_by_hk, eligible = orchestrator.calculate_weights(eligible, score, burn, BASE_HK)

    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in ENVS]
        + [f"L{s}" for s in range(1, N_envs + 1)]
        + ["Pts", "Elig", "FirstBlk", "Wgt"]
    )

    # Use shared row creation function (note: model_name_max_len=30 for this case)
    ranked_rows = sorted(
        (r for r in (
            _create_miner_row(
                hk, prev, weight_by_hk, acc, cnt, confidence_intervals,
                env_winners, layer_points, score, eligible, first_block,
                ENVS, N_envs, model_name_max_len=30
            ) for hk in eligible
        ) if r is not None),
        key=lambda r: float(r[-4]),
        reverse=True
    )

    unranked_rows = sorted(
        (r for r in (
            _create_miner_row(
                hk, prev, weight_by_hk, acc, cnt, confidence_intervals,
                env_winners, layer_points, score, eligible, first_block,
                ENVS, N_envs, model_name_max_len=30
            ) for hk in active_hks if hk not in eligible
        ) if r is not None),
        key=lambda r: float(r[-4]),
        reverse=True
    )

    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"), flush=True)

    # Save summary to S3
    ctx = SummaryContext(
        block=blk,
        header=hdr,
        rows=rows,
        eligible=eligible,
        active_hotkeys=active_hks,
        queryable_hotkeys=queryable_hks,
        env_winners=env_winners,
        accuracies=acc,
        counts=cnt,
        confidence_intervals=confidence_intervals,
        scores=score,
        layer_points=layer_points,
        first_blocks=first_block,
        metagraph=meta,
        previous_miners=prev
    )
    summary_data = _build_summary_data(ctx)
    if save_to_s3:
        await save_summary(blk, summary_data)
    else:
        logger.info("Skipping save to S3 (save_to_s3=False)")

    uids = [meta.hotkeys.index(hk) for hk in eligible]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights
