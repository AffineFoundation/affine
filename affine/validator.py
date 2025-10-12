import os
import time
import socket
import asyncio
import logging
import aiohttp
import traceback
import bittensor as bt
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse
from aiohttp import ClientConnectorError
from tabulate import tabulate
from .query import _get_client
from .storage import prune, CACHE_DIR

logger = logging.getLogger("affine")

async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "",
) -> bool:
    from .storage import get_conf
    from .utils.subtensor import get_subtensor
    
    for attempt in range(retries):
        try:
            st = await get_subtensor()
            ref = await st.get_current_block()
            logger.info(f"{log_prefix} set_weights attempt {attempt+1}/{retries}: netuid={netuid} uids={uids} weights={weights}")
            start = time.monotonic()
            bt.subtensor(get_conf('SUBTENSOR_ENDPOINT', default='finney')).set_weights(
                wallet=wallet, netuid=netuid, weights=weights, uids=uids,
                wait_for_inclusion=wait_for_inclusion,
            )
            logger.info(f"{log_prefix} extrinsic submitted in {(time.monotonic()-start)*1000:.1f}ms; waiting next block … (ref_block={ref})")
            await st.wait_for_block()
            meta = await st.metagraph(netuid)
            try:
                idx = meta.hotkeys.index(wallet.hotkey.ss58_address)
                lu = meta.last_update[idx]
                logger.info(f"{log_prefix} last_update={lu}, ref_block={ref}")
                if lu >= ref:
                    logger.info(f"{log_prefix} confirmation OK (last_update >= ref)")
                    return True
                logger.warning(f"{log_prefix} confirmation not yet included (last_update < ref), retrying …")
            except ValueError:
                logger.warning(f"{log_prefix} wallet hotkey not found in metagraph hotkeys; retrying …")
        except Exception as e:
            logger.warning(f"{log_prefix} set_weights attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}")
        await asyncio.sleep(delay_s)
    return False

async def retry_set_weights(wallet: bt.Wallet, uids: List[int], weights: List[float], retry: int = 10, netuid: int = None, lastset_metric = None):
    from .storage import get_conf
    from . import NETUID
    
    if netuid is None:
        netuid = NETUID
    
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
        timeout = aiohttp.ClientTimeout(connect=2, total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.monotonic()
            resp = await session.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": netuid,
                    "weights": weights,
                    "uids": uids,
                    "wait_for_inclusion": False,
                },
            )
            dur_ms = (time.monotonic() - start) * 1000
            logger.info(f"Signer HTTP response status={resp.status} in {dur_ms:.1f}ms")
            try:
                data = await resp.json()
            except Exception:
                txt = await resp.text()
                data = {"raw": (txt[:500] + ('…' if len(txt) > 500 else ''))}
            logger.info(f"Signer response body={data}")
            if resp.status == 200 and data.get("success"):
                if lastset_metric:
                    lastset_metric.set(time.time())
                return
            logger.warning(f"Signer responded error: status={resp.status} body={data}")
            return
    except ClientConnectorError as e:
        logger.info(f"Signer not reachable ({type(e).__name__}: {e}); falling back to local set_weights once")
        ok = await _set_weights_with_confirmation(
            wallet, netuid, uids, weights, False,
            retries=int(os.getenv("SIGNER_RETRIES", "10")),
            delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
            log_prefix="[validator-fallback]",
        )
        if ok:
            if lastset_metric:
                lastset_metric.set(time.time())
        else:
            logger.error("Local set_weights confirmation failed")
        return
    except asyncio.TimeoutError as e:
        logger.warning(f"Signer call timed out: {e}. Not falling back to local because validator has no wallet.")
        return

async def get_weights(tail: int = None, scale: float = 1, burn: float = 0.0, envs_tuple = None, netuid: int = None):
    from .sampling import MinerSampler, SamplingOrchestrator, SamplingConfig
    from .storage import dataset
    from .miners import miners
    from .utils.subtensor import get_subtensor
    from . import NETUID, ENVS
    
    if tail is None:
        tail = SamplingConfig.TAIL
    if envs_tuple is None:
        envs_tuple = ENVS
    if netuid is None:
        netuid = NETUID
    
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

    meta = await st.metagraph(netuid)
    BASE_HK = meta.hotkeys[0]
    N_envs = len(envs_tuple)
    
    queryable_miners = await miners(meta=meta, netuid=netuid)
    queryable_hks = {m.hotkey for m in queryable_miners.values()}
    logger.info(f"Found {len(queryable_hks)} queryable miners (hot, valid chute, not gated)")

    results_list = []
    
    initial_first_block = {}
    try:
        commits = await st.get_all_revealed_commitments(netuid)
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
    
    cnt, succ, prev, v_id, first_block = orchestrator.process_sample_data(
        results_list, meta.hotkeys, envs_tuple, BASE_HK
    )
    
    for hk, blk_val in initial_first_block.items():
        if hk not in first_block:
            first_block[hk] = blk_val
    
    acc = orchestrator.calculate_accuracies(cnt, succ, meta.hotkeys, envs_tuple)
    
    active_hks = list(prev.keys())
    logger.info("Computed accuracy & updated MAXENV.")

    eligible, required = sampler.calculate_eligibility(cnt, active_hks, queryable_hks, envs_tuple)
    logger.info(f"Eligible miners: {len(eligible)} (from {len(active_hks)} active, {len(queryable_hks)} queryable)")

    pool_for_dom = eligible if eligible else (queryable_hks & set(active_hks))
    
    dom_full = sampler.compute_dominance_counts(pool_for_dom, envs_tuple, acc, cnt, first_block)
    logger.info("Computed ε-dominance counts (full env set).")

    def ts(hk: str) -> int:
        return int(first_block[hk]) if hk in first_block else float('inf')
    
    best_candidates = pool_for_dom if pool_for_dom else (queryable_hks if queryable_hks else active_hks[:1])
    best = max(best_candidates, key=lambda hk: (dom_full.get(hk, 0), -ts(hk))) if best_candidates else active_hks[0]
    best_uid = meta.hotkeys.index(best)

    score, layer_points, env_winners = sampler.calculate_combinatoric_scores(
        envs_tuple, pool_for_dom, acc, cnt, first_block, scale
    )

    if not eligible:
        logger.warning(f"No eligible miners (queryable={len(queryable_hks)}); assigning weight 1.0 to uid 0.")
        
        hdr = (
            ["UID", "Model", "Rev"]
            + [f"{e}" for e in envs_tuple]
            + [f"L{s}" for s in range(1, N_envs + 1)]
            + ["Pts", "Elig", "Wgt"]
        )
        def row(hk: str):
            if hk not in prev:
                return None
            m = prev[hk].miner
            w = 1.0 if hk == best else 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for e in envs_tuple:
                base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}"
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
                f"{w:.4f}",
            ]
        rows = sorted((r for r in (row(hk) for hk in active_hks) if r is not None), key=lambda r: (r[-3], r[0]), reverse=True)
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))
        return [0], [1.0]

    weight_by_hk, eligible = orchestrator.calculate_weights(eligible, score, burn, BASE_HK)

    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in envs_tuple]
        + [f"L{s}" for s in range(1, N_envs + 1)]
        + ["Pts", "Elig", "Wgt"]
    )
    def row(hk: str):
        if hk not in prev:
            return None
        m = prev[hk].miner
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for e in envs_tuple:
            base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}"
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
            f"{w:.4f}",
        ]
    ranked_rows   = sorted((r for r in (row(hk) for hk in eligible) if r is not None), key=lambda r: float(r[-3]), reverse=True)
    unranked_rows = sorted((r for r in (row(hk) for hk in active_hks if hk not in eligible) if r is not None), key=lambda r: float(r[-3]), reverse=True)
    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    eligible_uids = [meta.hotkeys.index(hk) for hk in eligible]
    uids = [u for u in eligible_uids if u != best_uid] + [best_uid]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights

async def validate_api_key(api_key: str, base_url: str = "https://llm.chutes.ai/v1") -> bool:
    if not api_key:
        return False
    try:
        sess = await _get_client()
        async with sess.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=aiohttp.ClientTimeout(total=10.0)
        ) as response:
            return response.status >= 200 and response.status < 300
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False

def check_env_variables() -> bool:
    errors = []

    chutes_api_key = os.getenv("CHUTES_API_KEY", "")
    if not chutes_api_key:
        errors.append("CHUTES_API_KEY is not set")

    hf_user = os.getenv("HF_USER", "")
    if not hf_user:
        errors.append("HF_USER is not set")
    elif hf_user == "myaccount":
        errors.append("HF_USER is still set to default value 'myaccount'. Please set your actual Hugging Face username")

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        errors.append("HF_TOKEN is not set")
    elif hf_token == "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
        errors.append("HF_TOKEN is still set to default value. Please set your actual Hugging Face token")

    if errors:
        logger.error("Environment variable check failed:")
        for error in errors:
            logger.error(f"  - {error}")
        logger.info("\nPlease set the required environment variables in your .env file:")
        logger.info("  HF_USER=your_huggingface_username")
        logger.info("  HF_TOKEN=your_huggingface_token")
        return False
    
    return True