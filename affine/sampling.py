"""
Sampling module for Affine validator.
Handles miner sampling and weight calculation using epsilon-Pareto dominance.
"""

import math
import time
import logging
import asyncio
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set

logger = logging.getLogger("affine.sampling")


class SamplingConfig:
    """Configuration parameters for sampling."""
    
    TAIL = 20_000
    ALPHA = 0.9
    EPS_FLOOR = 0.005
    Z_NOT_WORSE = 1.28
    EPS_WIN = 0.015
    Z_WIN = 0.5
    ELIG = 0.10
    MIN_SAMPLES_PER_ENV = 100
    MAX_SAMPLES_CAP = 2000
    SAMPLE_COUNT_CAP = 1000


class MinerSampler:
    """Handles miner sampling and weight calculation."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        
    def calculate_eligibility(
        self,
        cnt: Dict[str, Dict[str, int]],
        active_hks: List[str],
        queryable_hks: Set[str],
        envs: Tuple[str, ...]
    ) -> Tuple[Set[str], Dict[str, int]]:
        """
        Determine eligible miners based on sample requirements.
        
        Returns:
            Tuple of (eligible_hotkeys, required_samples_per_env)
        """
        required = {}
        
        for e in envs:
            max_cnt = max((cnt[hk][e] for hk in active_hks), default=0)
            max_cnt = min(max_cnt, self.config.MAX_SAMPLES_CAP)
            required[e] = max(
                self.config.MIN_SAMPLES_PER_ENV, 
                10 + int(self.config.ELIG * max_cnt)
            )
        
        eligible = {
            hk for hk in active_hks 
            if hk in queryable_hks and all(cnt[hk][e] >= required[e] for e in envs)
        }
        
        return eligible, required
    
    def threshold_not_worse(
        self, 
        a_i: float, 
        n_i: int, 
        a_j: float, 
        n_j: int
    ) -> float:
        """
        Calculate tolerance for 'not worse' comparison on an environment.
        Uses statistical error with sample count capping.
        """
        if self.config.Z_NOT_WORSE <= 0:
            return self.config.EPS_FLOOR
            
        n_i_eff = min(int(n_i), self.config.SAMPLE_COUNT_CAP)
        n_j_eff = min(int(n_j), self.config.SAMPLE_COUNT_CAP)
        
        p_i = min(max(a_i, 0.0), 1.0)
        p_j = min(max(a_j, 0.0), 1.0)
        
        var = (p_i * (1 - p_i)) / max(n_i_eff, 1) + (p_j * (1 - p_j)) / max(n_j_eff, 1)
        return max(self.config.EPS_FLOOR, self.config.Z_NOT_WORSE * math.sqrt(max(var, 0.0)))
    
    def threshold_better(
        self,
        a_i: float,
        n_i: int,
        a_j: float,
        n_j: int,
        nw: float
    ) -> float:
        """
        Calculate margin to claim 'better on at least one environment'.
        Kept <= 'not worse' tolerance.
        """
        if self.config.Z_WIN > 0:
            n_i_eff = min(int(n_i), self.config.SAMPLE_COUNT_CAP)
            n_j_eff = min(int(n_j), self.config.SAMPLE_COUNT_CAP)
            
            p_i = min(max(a_i, 0.0), 1.0)
            p_j = min(max(a_j, 0.0), 1.0)
            
            var = (p_i * (1 - p_i)) / max(n_i_eff, 1) + (p_j * (1 - p_j)) / max(n_j_eff, 1)
            t = max(self.config.EPS_WIN, self.config.Z_WIN * math.sqrt(max(var, 0.0)))
        else:
            t = self.config.EPS_WIN
            
        return min(t, nw)
    
    def dominates_on(
        self,
        a: str,
        b: str,
        subset: Tuple[str, ...],
        acc: Dict[str, Dict[str, float]],
        cnt: Dict[str, Dict[str, int]],
        first_block: Dict[str, int]
    ) -> bool:
        """
        Check if miner 'a' dominates miner 'b' on the given environment subset.
        Uses epsilon-Pareto dominance with tie-breaking by earliest block.
        """
        not_worse_all = True
        better_any = False
        tie_all = True
        
        for e in subset:
            ai, aj = acc[a][e], acc[b][e]
            ni, nj = cnt[a][e], cnt[b][e]
            nw = self.threshold_not_worse(ai, ni, aj, nj)
            bet = self.threshold_better(ai, ni, aj, nj, nw)
            
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
    
    def compute_dominance_counts(
        self,
        pool: Set[str],
        envs: Tuple[str, ...],
        acc: Dict[str, Dict[str, float]],
        cnt: Dict[str, Dict[str, int]],
        first_block: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Compute dominance counts for the given miner pool on all environments.
        """
        dom_counts = defaultdict(int)
        
        for a, b in itertools.permutations(pool, 2):
            if self.dominates_on(a, b, envs, acc, cnt, first_block):
                dom_counts[a] += 1
                
        return dom_counts
    
    def find_subset_winner(
        self,
        env_subset: Tuple[str, ...],
        pool: Set[str],
        acc: Dict[str, Dict[str, float]],
        cnt: Dict[str, Dict[str, int]],
        first_block: Dict[str, int]
    ) -> Optional[str]:
        """
        Find winner on environment subset via epsilon-Pareto dominance.
        Falls back to earliest version start block if no dominance edges.
        """
        if not pool:
            return None
            
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool, 2):
            if self.dominates_on(x, y, env_subset, acc, cnt, first_block):
                dom_local[x] += 1
        
        def timestamp(hk: str) -> int:
            return int(first_block[hk]) if hk in first_block else float('inf')
        
        return max(pool, key=lambda hk: (dom_local.get(hk, 0), -timestamp(hk)))
    
    def compute_layer_weights(self, n_envs: int, scale: float) -> Dict[int, float]:
        """
        Compute per-subset weights K_s.
        K_1 = scale; K_s = scale * (2^s) for s >= 2.
        """
        K = {1: scale}
        for s in range(2, n_envs + 1):
            K[s] = scale * (2**s)
        return K
    
    def calculate_combinatoric_scores(
        self,
        envs: Tuple[str, ...],
        pool: Set[str],
        acc: Dict[str, Dict[str, float]],
        cnt: Dict[str, Dict[str, int]],
        first_block: Dict[str, int],
        scale: float
    ) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Dict[str, str]]:
        """
        Calculate combinatoric scores for all miners using all environment subsets.
        
        Returns:
            Tuple of (scores, layer_points, env_winners)
        """
        n_envs = len(envs)
        K = self.compute_layer_weights(n_envs, scale)
        
        scores = defaultdict(float)
        layer_points = defaultdict(lambda: defaultdict(float))
        env_winners = {}
        
        # Find single-env winners
        for e in envs:
            env_winners[e] = self.find_subset_winner((e,), pool, acc, cnt, first_block)
        
        # Award K_s to each subset winner
        for s in range(1, n_envs + 1):
            for env_subset in itertools.combinations(envs, s):
                winner = self.find_subset_winner(env_subset, pool, acc, cnt, first_block)
                if winner:
                    scores[winner] += K[s]
                    layer_points[winner][s] += K[s]
        
        return scores, layer_points, env_winners
    
    def apply_burn(
        self,
        weight_by_hk: Dict[str, float],
        burn: float,
        base_hotkey: str,
        eligible: Set[str]
    ) -> Tuple[Dict[str, float], Set[str]]:
        """
        Apply burn mechanism to weights.
        Scales eligible weights to (1 - burn) and assigns burn to base_hotkey.
        """
        if burn <= 0.0:
            return weight_by_hk, eligible
            
        burn = min(1.0, burn)
        
        # Scale existing eligible weights
        for hk in list(weight_by_hk.keys()):
            weight_by_hk[hk] = weight_by_hk.get(hk, 0.0) * (1.0 - burn)
        
        # Assign burn to base hotkey
        if base_hotkey in weight_by_hk:
            weight_by_hk[base_hotkey] += burn
        else:
            weight_by_hk[base_hotkey] = burn
            eligible = set(eligible)
            eligible.add(base_hotkey)
            
        return weight_by_hk, eligible


class SamplingOrchestrator:
    """Orchestrates the entire sampling and weight calculation process."""
    
    def __init__(self, sampler: Optional[MinerSampler] = None):
        self.sampler = sampler or MinerSampler()
    
    def process_sample_data(
        self,
        results: List[Any],
        meta_hotkeys: List[str],
        envs: Tuple[str, ...],
        base_hk: str
    ) -> Tuple[
        Dict[str, Dict[str, int]],
        Dict[str, Dict[str, int]],
        Dict[str, Any],
        Dict[str, Tuple[str, Optional[str]]],
        Dict[str, int]
    ]:
        """
        Process raw sample data into structured counts and metadata.
        
        Returns:
            Tuple of (cnt, succ, prev, v_id, first_block)
        """
        cnt = {hk: defaultdict(int) for hk in meta_hotkeys}
        succ = {hk: defaultdict(int) for hk in meta_hotkeys}
        prev = {}
        v_id = {}
        first_block = {}
        
        for result in results:
            hk = result.miner.hotkey
            env = result.challenge.env.name
            
            if hk not in cnt:
                continue
                
            # Check model family for non-base miners
            try:
                name = result.miner.model.split("/", 1)[1].lower()
            except Exception:
                name = str(result.miner.model).lower()
                
            if hk != base_hk and not name.startswith("affine"):
                continue
            
            cur_vid = (result.miner.model, result.miner.revision)
            
            # Handle version changes
            if v_id.get(hk) != cur_vid:
                v_id[hk] = cur_vid
                first_block[hk] = result.miner.block
                for e in envs:
                    cnt[hk][e] = 0
                    succ[hk][e] = 0
            else:
                # Update earliest block
                try:
                    fb = int(first_block.get(hk, result.miner.block)) if first_block.get(hk) is not None else int(result.miner.block)
                    cb = int(result.miner.block) if result.miner.block is not None else fb
                    first_block[hk] = fb if fb <= cb else cb
                except Exception:
                    pass
            
            prev[hk] = result
            if result.response.success:
                cnt[hk][env] += 1
                succ[hk][env] += float(result.evaluation.score)
                
        return cnt, succ, prev, v_id, first_block
    
    def calculate_accuracies(
        self,
        cnt: Dict[str, Dict[str, int]],
        succ: Dict[str, Dict[str, int]],
        meta_hotkeys: List[str],
        envs: Tuple[str, ...]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy scores for all miners across all environments."""
        acc = {}
        
        for hk in meta_hotkeys:
            acc[hk] = {}
            for e in envs:
                if cnt[hk][e] > 0:
                    acc[hk][e] = succ[hk][e] / cnt[hk][e]
                else:
                    acc[hk][e] = 0.0
                    
        return acc
    
    def calculate_weights(
        self,
        eligible: Set[str],
        scores: Dict[str, float],
        burn: float,
        base_hotkey: str
    ) -> Tuple[Dict[str, float], Set[str]]:
        """
        Calculate normalized weights from scores.
        
        Returns:
            Tuple of (weight_by_hk, updated_eligible)
        """
        if not eligible:
            return {}, eligible
            
        total_points = sum(scores.get(hk, 0.0) for hk in eligible)
        
        if total_points <= 0:
            # Fall back to best miner
            best = max(eligible, key=lambda hk: scores.get(hk, 0.0))
            weight_by_hk = {hk: (1.0 if hk == best else 0.0) for hk in eligible}
        else:
            weight_by_hk = {hk: (scores.get(hk, 0.0) / total_points) for hk in eligible}
        
        # Apply burn if requested
        if burn > 0:
            weight_by_hk, eligible = self.sampler.apply_burn(
                weight_by_hk, burn, base_hotkey, eligible
            )
            
        return weight_by_hk, eligible