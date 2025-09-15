
#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import math
import time
import asyncio
import traceback
import itertools
import bittensor as bt
from tabulate import tabulate
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable
import affine as af

# --- Scoring hyperparameters --------------------------------------------------
TAIL = 20_000
ALPHA = 0.9

# Tuned ε-margins:
#  - 'not-worse' uses a smaller Z to ease dominance when sample sizes are large.
#  - 'better_any' uses a tiny fixed margin so small but consistent edges can win size-1 subsets.
EPS_FLOOR   = 0.005    # 0.20 percentage points floor for "not worse" tolerance
Z_NOT_WORSE = 1.28     # one-sided ~80% cushion for "not worse" (was 1.645)
EPS_WIN     = 0.008  # 0.15 percentage points to claim "better on at least one env"
Z_WIN       = 0.5      # keep "better" threshold floor-based (set >0 to scale with n)
ELIG        = 0.03 

async def get_weights(tail: int = TAIL, scale: float = 1):
    """
    Compute miner weights using ε-Pareto dominance and combinatoric subset winners.

    Pipeline
      1) Ingest last `tail` blocks → per-miner per-env mean GRPO score.
      2) Determine eligibility (>=90% of per-env max total samples with scores).
      3) Global ε-dominance (all envs) for canonical 'best' (for tie breaks / summaries).
      4) Combinatoric scoring:
           - For every non-empty subset S of ENVS, pick the ε-Pareto winner on S.
           - Award K_|S| where K_1 = scale, K_s = C(N, s-1)*K_{s-1}.
         Fallback if no dominance edges on S: earliest version (earlier block wins).
      5) Normalize scores over eligibles to produce weights. Metrics + summary emitted.

    Returns:
      (uids, weights): list of eligible UIDs (best last) and their weights (sum to 1).
    """

    # --- fetch + prune --------------------------------------------------------
    st = await af.get_subtensor()
    meta = await st.metagraph(af.NETUID)
    BASE_HK = meta.hotkeys[0]
    N_envs = len(af.ENVS)
    
    # Tallies for all known hotkeys (so metrics update is safe even if some have no data)
    cnt      = {hk: defaultdict(int)     for hk in meta.hotkeys}  # per-env total samples with scores
    sumscore = {hk: defaultdict(float)   for hk in meta.hotkeys}  # per-env sum of GRPO scores
    sumsq    = {hk: defaultdict(float)   for hk in meta.hotkeys}  # per-env sum of squares
    current_miners = await af.get_miners(meta=meta)
    prev  = { m.hotkey: m for m in current_miners.values() }
    first_block = { m.hotkey: m.block for m in current_miners.values() }  # earliest block for current version
    pairs = [ (mi.hotkey, mi.revision) for mi in current_miners.values() ]
    # Parallelize per-env aggregation to speed up reads
    async def env_worker(env) -> Tuple[str, Dict[str, Dict[str, float]]]:
        try:
            env_cls = af.ENVS.get(str(env)) if hasattr(af, "ENVS") else None
            env_name = str(env)
            env_version = getattr(env_cls, "__version__", None)
            af.logger.info(f"weights: aggregating env={env_name} version={env_version} miners={len(pairs)}")
            t0 = time.perf_counter()
            agg = await af.aggregate_scores_by_env(env_name=env_name, pairs=pairs, env_version=env_version)
            af.logger.info(f"weights: aggregated env={env_name} in {time.perf_counter()-t0:.3f}s counts={ {hk: int((v or {}).get('n_total',0)) for hk,v in (agg or {}).items()} }")
            return env_name, (agg or {})
        except Exception as e:
            af.logger.warning(f'Error in dataset polling (agg) for env {env}... {e}')
            return str(env), {}

    tasks = [asyncio.create_task(env_worker(env)) for env in af.ENVS]
    for t in tasks:
        env_name, agg = await t
        try:
            total = 0
            for hk, stats in agg.items():
                n   = int(stats.get("n_total", 0) or 0)
                s   = float(stats.get("sum_score", 0.0) or 0.0)
                ssq = float(stats.get("sum_sq_score", 0.0) or 0.0)
                if n:
                    cnt[hk][env_name]      += n
                    sumscore[hk][env_name] += s
                    sumsq[hk][env_name]    += ssq
                    total += n
            af.logger.trace(f'Aggregated {total} total samples for env: {env_name}')
        except Exception as e:
            af.logger.warning(f'Error in dataset polling (agg) for env {env_name}... {e}')
    # --- mean GRPO + MAXENV ---------------------------------------------------
    mean = {
        hk: {e: (sumscore[hk][e] / cnt[hk][e] if cnt[hk][e] else 0.0) for e in af.ENVS}
        for hk in meta.hotkeys
    }

    active_hks = list(prev.keys())
    for e in af.ENVS:
        max_e = max((mean[hk][e] for hk in active_hks), default=0.0)
        af.MAXENV.labels(env=e).set(max_e)
    af.logger.info("Computed mean GRPO & updated MAXENV.")

    # --- eligibility: require near-max samples per env ------------------------
    required = {
        e: 10 + int(ELIG * max((cnt[hk][e] for hk in active_hks), default=0))
        for e in af.ENVS
    }
    eligible = {hk for hk in active_hks if all(cnt[hk][e] >= required[e] for e in af.ENVS)}

    # --- ε-Pareto dominance helpers ------------------------------------------
    def _var(hk: str, e: str) -> float:
        n = cnt[hk][e]
        if n <= 1:
            return 0.0
        s = sumscore[hk][e]
        ssq = sumsq[hk][e]
        mu = s / n
        v = (ssq / n) - (mu * mu)
        return v if v > 0.0 else 0.0

    def thr_not_worse(a_i: float, n_i: int, v_i: float, a_j: float, n_j: int, v_j: float) -> float:
        if Z_NOT_WORSE <= 0:
            return EPS_FLOOR
        se = math.sqrt((v_i / max(n_i, 1)) + (v_j / max(n_j, 1)))
        return max(EPS_FLOOR, Z_NOT_WORSE * se)

    def thr_better(a_i: float, n_i: int, v_i: float, a_j: float, n_j: int, v_j: float, nw: float) -> float:
        if Z_WIN > 0:
            se = math.sqrt((v_i / max(n_i, 1)) + (v_j / max(n_j, 1)))
            t = max(EPS_WIN, Z_WIN * se)
        else:
            t = EPS_WIN
        return min(t, nw)

    def dominates_on(a: str, b: str, subset) -> bool:
        """
        True iff 'a' is not-worse than 'b' on every env in `subset` (within thr_not_worse),
        and strictly better on at least one env by thr_better. Full ε-ties break by earlier start.
        """
        not_worse_all = True
        better_any    = False
        tie_all       = True
        for e in subset:
            ai, aj = mean[a][e], mean[b][e]
            ni, nj = cnt[a][e], cnt[b][e]
            vi, vj = _var(a, e), _var(b, e)
            nw  = thr_not_worse(ai, ni, vi, aj, nj, vj)
            bet = thr_better(ai, ni, vi, aj, nj, vj, nw)

            if ai < aj - nw:
                not_worse_all = False
            if ai >= aj + bet:
                better_any = True
            if abs(ai - aj) > nw:
                tie_all = False

        if not_worse_all and better_any:
            return True
        if tie_all and first_block.get(a, float("inf")) < first_block.get(b, float("inf")):
            return True
        return False

    # Global dominance (full ENVS) for summary + canonical "best"
    dom_full = defaultdict(int)
    pool_for_dom = eligible if eligible else set(active_hks)
    for a, b in itertools.permutations(pool_for_dom, 2):
        if dominates_on(a, b, af.ENVS):
            dom_full[a] += 1
    af.logger.info("Computed ε-dominance counts (full env set).")

    def ts(hk: str) -> int:
        """Block-number timestamp; default to last seen block."""
        return int(first_block.get(hk, prev[hk].block))

    best = max(pool_for_dom, key=lambda hk: (dom_full.get(hk, 0), -ts(hk))) if pool_for_dom else active_hks[0]
    best_uid = meta.hotkeys.index(best)

    # --- combinatoric scoring over all non-empty env subsets ------------------
    def layer_weights(N: int, kappa: float):
        """Per-subset weights K_s: K_1=kappa; K_s=C(N,s-1)*K_{s-1} for s>=2."""
        K = {1: kappa}
        for s in range(2, N + 1):
            K[s] = kappa * (2**s)
        return K

    def subset_winner(env_subset):
        """
        Winner on env_subset via ε-Pareto. If no dominance edges, fall back to:
          earliest version start block (earlier block wins).
        """
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool_for_dom, 2):
            if dominates_on(x, y, env_subset):
                dom_local[x] += 1

        return max(pool_for_dom, key=lambda hk: (dom_local.get(hk, 0), -ts(hk)))

    # Calculate combinatoric scores for all miners (not just eligible)
    K = layer_weights(N_envs, scale)
    score = defaultdict(float)
    layer_points = {hk: defaultdict(float) for hk in active_hks}

    # --- Find single-env winners for highlighting ----------------------------
    env_winners = {}
    for e in af.ENVS:
        env_winners[e] = subset_winner((e,))

    # Award K_s to each subset winner
    for s in range(1, N_envs + 1):
        for env_subset in itertools.combinations(af.ENVS, s):
            w = subset_winner(env_subset)
            score[w] += K[s]
            layer_points[w][s] += K[s]

    # If no eligible miners exist, fall back to the canonical best with weight 1.0.
    if not eligible:
        af.logger.warning("No eligible miners; assigning weight 1.0 to canonical best.")
        for uid, hk in enumerate(meta.hotkeys):
            af.WEIGHT.labels(uid=uid).set(1.0 if hk == best else 0.0)
            for e in af.ENVS:
                a = mean[hk][e]
                if cnt[hk][e] > 0:
                    af.SCORE.labels(uid=uid, env=e).set(a)

        hdr = (
            ["UID", "Model", "Rev", "BLK"]
            + [f"{e}" for e in af.ENVS]
            + [f"L{s}" for s in range(1, N_envs + 1)]
            + ["Pts", "Elig", "Wgt"]
        )
        def row(hk: str):
            m = prev[hk]
            w = 1.0 if hk == best else 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for e in af.ENVS:
                base = f"{100 * mean[hk][e]:.2f}/{cnt[hk][e]}"
                if hk == env_winners.get(e):
                    env_cols.append(f"*{base}*")
                else:
                    env_cols.append(base)
            return [
                m.uid, model_name, str(m.revision)[:5], m.block,
                *env_cols,
                *[f"{layer_points[hk][s]:.1f}" for s in range(1, N_envs + 1)],
                f"{score.get(hk, 0.0):.2f}",
                "Y" if hk in eligible else "N",
                f"{w:.4f}",
            ]
        rows = sorted((row(hk) for hk in active_hks), key=lambda r: (r[-3], r[0]), reverse=True)
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))
        return [best_uid], [1.0]

    # Eligible path: normalize scores to weights over the eligible pool only
    total_points = sum(score[hk] for hk in eligible)
    if total_points <= 0:
        af.logger.warning("Combinatoric scoring returned zero total; falling back to canonical best.")
        weight_by_hk = {hk: (1.0 if hk == best else 0.0) for hk in eligible}
    else:
        weight_by_hk = {hk: (score[hk] / total_points) for hk in eligible}

    # --- summary printout -----------------------------------------------------
    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in af.ENVS]
        + [f"L{s}" for s in range(1, N_envs + 1)]
        + ["Pts", "Elig", "Wgt"]
    )
    def row(hk: str):
        m = prev[hk]
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for e in af.ENVS:
            base = f"{100 * mean[hk][e]:.2f}/{cnt[hk][e]}"
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
    ranked_rows   = sorted((row(hk) for hk in eligible), key=lambda r: float(r[-3]), reverse=True)
    unranked_rows = sorted((row(hk) for hk in active_hks if hk not in eligible), key=lambda r: float(r[-3]), reverse=True)
    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    # --- Prometheus updates ---------------------------------------------------
    for uid, hk in enumerate(meta.hotkeys):
        af.WEIGHT.labels(uid=uid).set(weight_by_hk.get(hk, 0.0))
        for e in af.ENVS:
            a = mean[hk][e]
            if cnt[hk][e] > 0:
                af.SCORE.labels(uid=uid, env=e).set(a)

    # --- Return weights in a stable shape (best last, as before) -------------
    eligible_uids = [meta.hotkeys.index(hk) for hk in eligible]
    uids = [u for u in eligible_uids if u != best_uid] + [best_uid]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights


        
@af.cli.command("validate")
def validate():
    coldkey = af.get_conf("BT_WALLET_COLD", "default")
    hotkey  = af.get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)    
    async def _run():     
        LAST = 0
        TEMPO = 360
        INNER_TEMPO = 100
        NETUID = 120
        subtensor = None
        while True:
            try:
                # ---------------- Wait for set weights. -----------------
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
                if subtensor is None: subtensor = await af.get_subtensor()
                BLOCK = await subtensor.get_current_block()
                I = (TEMPO + 1 + NETUID + 1 + BLOCK) % (TEMPO + 1) % INNER_TEMPO
                if I != 0:
                    af.logger.debug(f'Waiting ... ({TEMPO} + 1 + {NETUID} + 1 + {BLOCK}) % ({TEMPO} + 1) % {INNER_TEMPO} = {I} != 0')
                    await subtensor.wait_for_block()
                    continue
                
                # ---------------- Set weights. ------------------------
                uids, weights = await get_weights()
        
                # ---------------- Set weights. ------------------------
                af.logger.info("Setting weights ...")
                await af.retry_set_weights( wallet, uids=uids, weights=weights, retry = 3)
                subtensor = await af.get_subtensor()
                SETBLOCK = await subtensor.get_current_block()
                af.LASTSET.set_function(lambda: SETBLOCK - LAST)
                LAST = BLOCK           
                            
            except asyncio.CancelledError: break
            except Exception as e:
                traceback.print_exc()
                af.logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
            
    async def main():
        await asyncio.gather(
            _run(),
            af.watchdog(timeout = (60 * 20))
        )
    asyncio.run(main())
    
    
@af.cli.command("weights")
@af.click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def weights(verbose: int):
    af.setup_logging(verbose)
    asyncio.run(get_weights())
