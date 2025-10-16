"""
Sampling module for Affine validator.
Handles miner sampling and weight calculation using z-score based Pareto dominance.
"""

import math
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set


class SamplingConfig:
    """Configuration parameters for sampling."""
    
    TAIL = 20_000
    ALPHA = 0.9
    ELIG = 0.10
    MIN_SAMPLES_PER_ENV = 100
    MAX_SAMPLES_CAP = 2000
    # Z-score threshold for significance on an environment (two-sided ~ 90% one-sided)
    Z_EPS = 1.29
    # Pareto rule: at least this many strictly better envs
    PARETO_MIN_BETTER = 2
    # Pareto rule: at most this many strictly worse envs (eligibility cutoff)
    PARETO_MAX_WORSE = 2


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
    
    def compute_env_stats(
        self,
        envs: Tuple[str, ...],
        pool: Set[str],
        acc: Dict[str, Dict[str, float]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute per-environment (mean, std) of accuracies across the pool.
        Returns a mapping env -> (mean, std).
        """
        stats = {}
        for e in envs:
            values = [acc[hk][e] for hk in pool]
            if not values:
                stats[e] = (0.0, 0.0)
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) * (v - mean) for v in values) / len(values)
            std = math.sqrt(var)
            stats[e] = (mean, std)
        return stats

    def compute_zscores(
        self,
        envs: Tuple[str, ...],
        pool: Set[str],
        acc: Dict[str, Dict[str, float]],
        env_stats: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-miner per-env z-scores using provided env_stats (mean, std).
        If std is zero, z-score is treated as 0.0 (tie).
        """
        zscores: Dict[str, Dict[str, float]] = {}
        for hk in pool:
            zscores[hk] = {}
            for e in envs:
                mean, std = env_stats.get(e, (0.0, 0.0))
                if std <= 0.0:
                    z = 0.0
                else:
                    z = (acc[hk][e] - mean) / std
                zscores[hk][e] = z
        return zscores
    
    def dominates_on(
        self,
        a: str,
        b: str,
        subset: Tuple[str, ...],
        acc: Dict[str, Dict[str, float]],
        env_stats: Dict[str, Tuple[float, float]],
        first_block: Dict[str, int]
    ) -> bool:
        """
        Check if miner 'a' dominates miner 'b' on the given environment subset.
        Uses z-score based Pareto dominance with tie-breaking by earliest block.
        """
        better_count = 0
        worse_count = 0
        tie_all = True
        
        for e in subset:
            mean, std = env_stats.get(e, (0.0, 0.0))
            # If std is zero, treat as tie (no significant variance)
            if std <= 0.0:
                continue
            z_diff = (acc[a][e] - acc[b][e]) / std
            if z_diff >= self.config.Z_EPS:
                better_count += 1
                tie_all = False
            elif z_diff <= -self.config.Z_EPS:
                worse_count += 1
                tie_all = False
        
        if better_count >= self.config.PARETO_MIN_BETTER and worse_count <= self.config.PARETO_MAX_WORSE:
            return True
        if tie_all and first_block.get(a, float("inf")) < first_block.get(b, float("inf")):
            return True
        return False
    
    def compute_dominance_counts(
        self,
        pool: Set[str],
        envs: Tuple[str, ...],
        acc: Dict[str, Dict[str, float]],
        env_stats: Dict[str, Tuple[float, float]],
        first_block: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Compute dominance counts for the given miner pool on all environments.
        """
        # Backward compatibility: if env_stats looks like counts or is empty, recompute
        try:
            any_val = next(iter(env_stats.values())) if env_stats else None
            if not (isinstance(any_val, tuple) and len(any_val) == 2 and all(isinstance(x, (int, float)) for x in any_val)):
                env_stats = self.compute_env_stats(envs, pool, acc)
        except Exception:
            env_stats = self.compute_env_stats(envs, pool, acc)

        dom_counts = defaultdict(int)
        
        for a, b in itertools.permutations(pool, 2):
            if self.dominates_on(a, b, envs, acc, env_stats, first_block):
                dom_counts[a] += 1
                
        return dom_counts
    
    # Note: subset winner and layered weights from the previous epsilon-Pareto
    # approach have been removed. Scoring is now per-env with a single-layer
    # point scheme and earliest-commit tie-breaks.
    
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
        Calculate scores via per-environment point assignment using z-scores.
        Rules:
          - A miner is ineligible if they are worse (z <= -Z_EPS) on 2 or more envs.
          - A miner is eligible only if they have at least 2 better envs (z >= Z_EPS).
          - For each env, award exactly 1 point to the earliest-commit miner among
            eligible miners with z >= Z_EPS. If none, no point is awarded.
          - Final weight is normalized from total points.
        Returns: (scores, layer_points, env_winners)
        """
        scores = defaultdict(float)
        layer_points = defaultdict(lambda: defaultdict(float))
        env_winners: Dict[str, str] = {}

        # Precompute env stats and z-scores
        env_stats = self.compute_env_stats(envs, pool, acc)
        zscores = self.compute_zscores(envs, pool, acc, env_stats)

        # Determine miner eligibility (filter only by number of "worse" envs)
        eligible: Set[str] = set()
        for hk in pool:
            worse = sum(1 for e in envs if zscores[hk][e] <= -self.config.Z_EPS)
            if worse <= self.config.PARETO_MAX_WORSE:
                eligible.add(hk)

        # Award 1 point per env:
        #   - Prefer earliest eligible miner with z >= Z_EPS on that env
        #   - If none, earliest eligible miner with z > -Z_EPS (not worse)
        #   - If still none, earliest eligible miner
        for e in envs:
            candidates = [hk for hk in pool if hk in eligible and zscores[hk][e] >= self.config.Z_EPS]
            if not candidates:
                candidates = [hk for hk in pool if hk in eligible and zscores[hk][e] > -self.config.Z_EPS]
            if not candidates:
                candidates = [hk for hk in pool if hk in eligible]
            if not candidates:
                continue
            winner = min(candidates, key=lambda hk: first_block.get(hk, float('inf')))
            env_winners[e] = winner
            scores[winner] += 1.0
            layer_points[winner][1] += 1.0

        # Global fallback: if no points at all, pick earliest among miners with <=1 worse env
        if sum(scores.values()) <= 0.0 and pool:
            fallback_candidates = [
                hk for hk in pool
                if sum(1 for e in envs if zscores[hk][e] <= -self.config.Z_EPS) <= 1
            ]
            if fallback_candidates:
                fb_winner = min(fallback_candidates, key=lambda hk: first_block.get(hk, float('inf')))
                scores[fb_winner] += 1.0
                layer_points[fb_winner][1] += 1.0

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
            env = result.challenge.env
            
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
            # Fallback: pick any best-by-score (should be rare since scoring handles ties)
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