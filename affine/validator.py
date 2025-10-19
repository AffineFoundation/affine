import os
import time
import socket
import asyncio
import logging
import aiohttp
import math
import itertools
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
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "",
) -> bool:
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
        timeout = aiohttp.ClientTimeout(connect=2, total=120)
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
            log_prefix="[validator-fallback]",
        )
        if not ok:
            logger.error("Local set_weights confirmation failed")
        return
    except asyncio.TimeoutError as e:
        logger.warning(f"Signer call timed out: {e}. Not falling back to local because validator has no wallet.")
        return

async def get_weights(tail: int = SamplingConfig.TAIL, scale: float = 1, burn: float = 0.0):
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
    
    queryable_miners = await miners(meta=meta, netuid=NETUID, check_validity=False)
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
    
    cnt, succ, prev, v_id, first_block, pairs_by_time = orchestrator.process_sample_data(
        results_list, meta.hotkeys, ENVS, BASE_HK
    )
    
    for hk, blk_val in initial_first_block.items():
        if hk not in first_block:
            first_block[hk] = blk_val
    
    acc = orchestrator.calculate_accuracies(cnt, succ, meta.hotkeys, ENVS)
    
    active_hks = list(prev.keys())
    logger.info("Computed accuracy & updated MAXENV.")

    eligible, required = sampler.calculate_eligibility(cnt, active_hks, queryable_hks, ENVS)
    logger.info(f"Eligible miners: {len(eligible)} (from {len(active_hks)} active, {len(queryable_hks)} queryable)")

    # MAE-based similarity grouping post-filter to prevent copier clusters
    # Build per-miner per-env stats using Welford from filtered results
    try:
        threshold = float(get_conf('AFFINE_MAE_THRESHOLD', default='0.20'))
    except Exception:
        threshold = 0.20

    stats = {}
    for hk in active_hks:
        stats[hk] = {e: {'n': 0, 'mean': 0.0, 'm2': 0.0} for e in ENVS}

    def _consider_result(r):
        # Mirror process_sample_data filter so stats align with scoring pipeline
        hk = r.miner.hotkey
        if hk not in stats:
            return False
        try:
            name = r.miner.model.split("/", 1)[1].lower()
        except Exception:
            name = str(r.miner.model).lower()
        if hk != BASE_HK and not name.startswith("affine"):
            return False
        return True

    for r in results_list:
        if not _consider_result(r):
            continue
        hk = r.miner.hotkey
        e = r.challenge.env
        if e not in ENVS:
            continue
        x = float(r.evaluation.score)
        s = stats[hk][e]
        n1 = s['n'] + 1
        delta = x - s['mean']
        mean1 = s['mean'] + delta / n1
        m2_1 = s['m2'] + delta * (x - mean1)
        s['n'], s['mean'], s['m2'] = n1, mean1, m2_1

    def _finalize(hk, e):
        s = stats[hk][e]
        n = s['n']
        if n <= 1:
            return n, s['mean'], 0.0
        return n, s['mean'], math.sqrt(max(0.0, s['m2'] / (n - 1)))

    def _directed_mae(a: str, b: str) -> float:
        vals = []
        for e in ENVS:
            na, ma, sa = _finalize(a, e)
            nb, mb, _ = _finalize(b, e)
            if na >= 2 and sa > 0.0 and nb >= 1:
                vals.append(abs(mb - ma) / sa)
        if not vals:
            return float('inf')
        return sum(vals) / len(vals)

    def _sym_mae(a: str, b: str) -> float:
        dab = _directed_mae(a, b)
        dba = _directed_mae(b, a)
        if math.isinf(dab) or math.isinf(dba):
            return float('inf')
        return 0.5 * (dab + dba)

    # Build similarity graph on current eligible set
    elig_list = list(eligible)
    graph = {hk: set() for hk in elig_list}
    for i in range(len(elig_list)):
        ai = elig_list[i]
        for j in range(i + 1, len(elig_list)):
            bj = elig_list[j]
            d = _sym_mae(ai, bj)
            if d < threshold:
                graph[ai].add(bj)
                graph[bj].add(ai)

    # Connected components = groups; keep earliest block in each group
    visited = set()
    group_winners = []
    groups = []
    for hk in elig_list:
        if hk in visited:
            continue
        comp = []
        stack = [hk]
        visited.add(hk)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in graph[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        groups.append(comp)
        winner = min(comp, key=lambda x: int(first_block.get(x, float('inf'))))
        group_winners.append(winner)

    if groups:
        logger.info("MAE similarity groups (threshold=%.3f):" % threshold)
        for idx, comp in enumerate(groups, 1):
            members = sorted(comp, key=lambda x: int(first_block.get(x, float('inf'))))
            earliest = members[0]
            uids = [meta.hotkeys.index(h) for h in members]
            logger.info(f"  Group {idx}: size={len(comp)} earliest={earliest} uids={uids}")

    eligible = set(group_winners) if eligible else eligible
    pool_for_dom = eligible if eligible else (queryable_hks & set(active_hks))
    
    elo = sampler.compute_elo_scores(pairs_by_time, meta.hotkeys, ENVS)
    logger.info(f"Computed ELO scores from {len(pairs_by_time)} pairwise comparisons.")
    
    epsilon_per_env = sampler.compute_epsilon_from_ranking_gap(elo, ENVS)
    logger.info(f"Computed per-environment epsilon for dominance:")
    for env_name, eps_val in epsilon_per_env.items():
        logger.info(f"  {env_name}: {eps_val:.2f}")
    
    score, layer_points, env_winners = sampler.calculate_combinatoric_scores(
        ENVS, pool_for_dom, elo, first_block, epsilon_per_env, scale
    )

    # Select best miner based on Nash score (no dominance)
    best_candidates = pool_for_dom if pool_for_dom else (queryable_hks if queryable_hks else active_hks[:1])
    best = max(best_candidates, key=lambda hk: score.get(hk, 0.0)) if best_candidates else active_hks[0]
    best_uid = meta.hotkeys.index(best)

    # Print final per-miner scores (uid, score) before the table
    try:
        lines = ["Final scores (Nash):"]
        for hk in active_hks:
            if hk in prev:
                uid = prev[hk].miner.uid
                s = score.get(hk, 0.0)
                lines.append(f"  uid={uid} score={s:.2f}")
        print("\n".join(lines))
    except Exception:
        pass

    if not eligible:
        logger.warning(f"No eligible miners (queryable={len(queryable_hks)}); assigning weight 1.0 to uid 0.")
        
        hdr = (
            ["UID", "Model", "Rev"]
            + [f"{e}" for e in ENVS]
            + ["Score", "Elig", "Wgt"]
        )
        def row(hk: str):
            if hk not in prev:
                return None
            m = prev[hk].miner
            w = 1.0 if hk == best else 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for e in ENVS:
                base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}/{elo[hk][e]:.0f}"
                if hk == env_winners.get(e):
                    env_cols.append(f"*{base}*")
                else:
                    env_cols.append(base)
            return [
                m.uid, model_name, str(m.revision)[:5],
                *env_cols,
                f"{score.get(hk, 0.0):.2f}",
                "Y" if hk in eligible else "N",
                f"{w:.4f}",
            ]
        # Print MAE groups summary (uids per group) before the table
        if groups:
            grp_lines = [f"MAE groups (uids per group, count={len(groups)}):"]
            for idx, comp in enumerate(groups, 1):
                ordered = sorted(comp, key=lambda x: int(first_block.get(x, float('inf'))))
                uids = [meta.hotkeys.index(h) for h in ordered]
                grp_lines.append(f"  G{idx}: uids={uids}")
            print("\n".join(grp_lines))

        rows = sorted((r for r in (row(hk) for hk in active_hks) if r is not None), key=lambda r: (r[-3], r[0]), reverse=True)
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))
        return [0], [1.0]

    weight_by_hk, eligible = orchestrator.calculate_weights(eligible, score, burn, BASE_HK)

    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in ENVS]
        + ["Score", "Elig", "Wgt"]
    )
    def row(hk: str):
        if hk not in prev:
            return None
        m = prev[hk].miner
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for e in ENVS:
            base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}/{elo[hk][e]:.0f}"
            if hk == env_winners.get(e):
                env_cols.append(f"*{base}*")
            else:
                env_cols.append(base)
        return [
            m.uid, model_name[:30], str(m.revision)[:5],
            *env_cols,
            f"{score.get(hk, 0.0):.2f}",
            "Y" if hk in eligible else "N",
            f"{w:.4f}",
        ]
    ranked_rows   = sorted((r for r in (row(hk) for hk in eligible) if r is not None), key=lambda r: float(r[-3]), reverse=True)
    unranked_rows = sorted((r for r in (row(hk) for hk in active_hks if hk not in eligible) if r is not None), key=lambda r: float(r[-3]), reverse=True)
    rows = ranked_rows + unranked_rows

    # Print MAE groups summary (uids per group) before the table
    if groups:
        grp_lines = [f"MAE groups (uids per group, count={len(groups)}):"]
        for idx, comp in enumerate(groups, 1):
            ordered = sorted(comp, key=lambda x: int(first_block.get(x, float('inf'))))
            uids = [meta.hotkeys.index(h) for h in ordered]
            grp_lines.append(f"  G{idx}: uids={uids}")
        print("\n".join(grp_lines))

    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    eligible_uids = [meta.hotkeys.index(hk) for hk in eligible]
    uids = [u for u in eligible_uids if u != best_uid] + [best_uid]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights
