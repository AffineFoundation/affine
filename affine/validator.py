
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
EPS_FLOOR   = 0.002    # 0.20 percentage points floor for "not worse" tolerance
Z_NOT_WORSE = 0.84     # one-sided ~80% cushion for "not worse" (was 1.645)
EPS_WIN     = 0.0015   # 0.15 percentage points to claim "better on at least one env"
Z_WIN       = 0.0      # keep "better" threshold floor-based (set >0 to scale with n)
ELIG        = 0.01 

async def get_weights(tail: int = TAIL, scale: float = 1):
    """
    Compute miner weights using ε-Pareto dominance and combinatoric subset winners.

    Pipeline
      1) Ingest last `tail` blocks → per-miner per-env accuracy.
      2) Determine eligibility (>=90% of per-env max count).
      3) Global ε-dominance (all envs) for canonical 'best' (for tie breaks / summaries).
      4) Combinatoric scoring:
           - For every non-empty subset S of ENVS, pick the ε-Pareto winner on S.
           - Award K_|S| where K_1 = scale, K_s = C(N, s-1)*K_{s-1}.
         Fallback if no dominance edges on S: highest mean accuracy on S, then earliest version.
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
    cnt   = {hk: defaultdict(int)   for hk in meta.hotkeys}  # per-env counts
    succ  = {hk: defaultdict(int)   for hk in meta.hotkeys}  # per-env correct (0/1 or [0,1])
    prev  = {}                                                # last sample per hk
    first_block = {}                                          # earliest block for current version
    current_miners = await af.get_miners(meta=meta)
    db_sem = asyncio.Semaphore(4)
    async def process_miner(uid, mi):
        if mi.hotkey not in cnt:
            return
        for env in af.ENVS:
            try:
                async with db_sem:
                    count = await af.count(env_name=str(env), hotkey=mi.hotkey, revision=mi.revision)
                    if count == 0: continue
                    rows = await af.select_rows(env_name=str(env), hotkey=mi.hotkey, revision=mi.revision)
                for r in rows:
                    if r['success']:
                        cnt[mi.hotkey][str(env)] += 1
                        succ[mi.hotkey][str(env)] += float(r['score'])
            except Exception as e:
                af.logger.warning(f'Error in dataset polling... {e}')
    await asyncio.gather(*(process_miner(uid, mi) for uid, mi in current_miners.items()))

    if not prev:
        af.logger.warning("No results collected; defaulting to uid 0")
        return [0], [1.0]

    # --- accuracy + MAXENV ----------------------------------------------------
    acc = {
        hk: {e: (succ[hk][e] / cnt[hk][e] if cnt[hk][e] else 0.0) for e in af.ENVS}
        for hk in meta.hotkeys
    }

    active_hks = list(prev.keys())
    for e in af.ENVS:
        max_e = max((acc[hk][e] for hk in active_hks), default=0.0)
        af.MAXENV.labels(env=e).set(max_e)
    af.logger.info("Computed accuracy & updated MAXENV.")

    # --- eligibility: require near-max samples per env ------------------------
    required = {
        e: int(ELIG * max((cnt[hk][e] for hk in active_hks), default=0))
        for e in af.ENVS
    }
    eligible = {hk for hk in active_hks if all(cnt[hk][e] >= required[e] for e in af.ENVS)}

    # --- ε-Pareto dominance helpers ------------------------------------------
    def thr_not_worse(a_i: float, n_i: int, a_j: float, n_j: int) -> float:
        """Tolerance for 'not worse' on an env: max(EPS_FLOOR, Z * SE_diff)."""
        if Z_NOT_WORSE <= 0:
            return EPS_FLOOR
        var = (a_i * (1 - a_i)) / max(n_i, 1) + (a_j * (1 - a_j)) / max(n_j, 1)
        return max(EPS_FLOOR, Z_NOT_WORSE * math.sqrt(var))

    def thr_better(a_i: float, n_i: int, a_j: float, n_j: int, nw: float) -> float:
        """
        Margin to claim 'better on at least one env'. Kept ≤ 'not worse' tolerance.
        Floor-based by default; set Z_WIN>0 to scale with SE_diff.
        """
        if Z_WIN > 0:
            var = (a_i * (1 - a_i)) / max(n_i, 1) + (a_j * (1 - a_j)) / max(n_j, 1)
            t = max(EPS_WIN, Z_WIN * math.sqrt(var))
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
            ai, aj = acc[a][e], acc[b][e]
            ni, nj = cnt[a][e], cnt[b][e]
            nw  = thr_not_worse(ai, ni, aj, nj)
            bet = thr_better(ai, ni, aj, nj, nw)

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
        return int(first_block.get(hk, prev[hk].miner.block))

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
          1) highest mean accuracy on the subset,
          2) earliest version start block.
        """
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool_for_dom, 2):
            if dominates_on(x, y, env_subset):
                dom_local[x] += 1

        def mean_acc(hk: str) -> float:
            return sum(acc[hk][e] for e in env_subset) / len(env_subset)

        return max(pool_for_dom, key=lambda hk: (dom_local.get(hk, 0), mean_acc(hk), -ts(hk)))

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
                a = acc[hk][e]
                if a > 0:
                    af.SCORE.labels(uid=uid, env=e).set(a)

        hdr = (
            ["UID", "Model", "Rev"]
            + [f"{e}" for e in af.ENVS]
            + [f"L{s}" for s in range(1, N_envs + 1)]
            + ["Pts", "Elig", "Wgt"]
        )
        def row(hk: str):
            m = prev[hk].miner
            w = 1.0 if hk == best else 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for e in af.ENVS:
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
        m = prev[hk].miner
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for e in af.ENVS:
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
    ranked_rows   = sorted((row(hk) for hk in eligible), key=lambda r: float(r[-3]), reverse=True)
    unranked_rows = sorted((row(hk) for hk in active_hks if hk not in eligible), key=lambda r: float(r[-3]), reverse=True)
    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    # --- Prometheus updates ---------------------------------------------------
    for uid, hk in enumerate(meta.hotkeys):
        af.WEIGHT.labels(uid=uid).set(weight_by_hk.get(hk, 0.0))
        for e in af.ENVS:
            a = acc[hk][e]
            if a > 0:
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
        TEMPO = 100
        subtensor = None
        while True:
            try:
                # ---------------- Wait for set weights. -----------------
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
                if subtensor is None: subtensor = await af.get_subtensor()
                BLOCK = await subtensor.get_current_block()
                if BLOCK % TEMPO != 0 or BLOCK <= LAST: 
                    af.logger.debug(f'Waiting ... {BLOCK} % {TEMPO} == {BLOCK % TEMPO} != 0')
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
            
                # ---------------- Other telemetry ------------------------
                af.CACHE.set(sum( f.stat().st_size for f in af.CACHE_DIR.glob("*.jsonl") if f.is_file()))
                
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
def weights():
    asyncio.run(get_weights())