"""
Sampling module for Affine validator.
Handles miner sampling and weight calculation using ELO-based per-environment margins
(signed Nash scoring) with MAE grouping and positive-score weight normalization.
"""

import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set


def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> Tuple[float, float]:
    """
    Update ELO ratings for two players.
    
    Args:
        rating_a: Current ELO rating of player A
        rating_b: Current ELO rating of player B
        score_a: Actual score of A (1.0 for win, 0.5 for draw, 0.0 for loss)
        k: K-factor for ELO update speed
    
    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1.0 - score_a) - expected_b)
    
    return new_rating_a, new_rating_b


class SamplingConfig:
    """Configuration parameters for sampling."""
    
    TAIL = 40000
    ALPHA = 0.9
    EPS_FLOOR = 0.005
    ELIG = 0.10
    MIN_SAMPLES_PER_ENV = 100
    MAX_SAMPLES_CAP = 2000
    SAMPLE_COUNT_CAP = 1000
    RANK_GAP = 2  # Use 3rd place (0-indexed) for epsilon calculation
    # Signed Nash scoring parameters (beta removed; use log1p on raw margins)
    NASH_GAMMA = 0.2   # penalty weight for negative margins (>1 penalizes losses more)
    # Epsilon adjustment for Nash
    NASH_EPS_SCALE = 0.0     # scale epsilon down to be less strict
    NASH_EPS_FLOOR_ELO = 0.0  # minimum epsilon in ELO units


class MinerSampler:
    """Handles miner sampling and weight calculation with ELO scoring."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        self.elo_k_factor = 32.0
    
    def compute_elo_scores(
        self,
        results_by_time: List[Tuple[Any, Any]],
        hotkeys: List[str],
        envs: Tuple[str, ...]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ELO scores for miners by processing pairwise results in time order.
        
        Args:
            results_by_time: List of (result_a, result_b) pairs in time order
            hotkeys: All hotkeys in metagraph
            envs: Environment names
            
        Returns:
            Dict mapping hotkey -> env -> elo_score
        """
        elo = {hk: {e: 1500.0 for e in envs} for hk in hotkeys}
        
        for result_a, result_b in results_by_time:
            hk_a = result_a.miner.hotkey
            hk_b = result_b.miner.hotkey
            env = result_a.challenge.env
            
            if env not in envs or hk_a not in elo or hk_b not in elo:
                continue
                
            score_a = float(result_a.evaluation.score) if result_a.response.success else 0.0
            score_b = float(result_b.evaluation.score) if result_b.response.success else 0.0
            
            if score_a > score_b:
                match_score = 1.0
            elif score_a < score_b:
                match_score = 0.0
            else:
                match_score = 0.5
            
            new_elo_a, new_elo_b = elo_update(
                elo[hk_a][env], elo[hk_b][env], match_score, self.elo_k_factor
            )
            elo[hk_a][env] = new_elo_a
            elo[hk_b][env] = new_elo_b
        
        return elo
        
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
    
    def compute_epsilon_from_ranking_gap(
        self,
        elo_rankings: Dict[str, Dict[str, float]],
        envs: Tuple[str, ...],
        rank_gap: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate per-environment epsilon based on ELO gap between top-ranked miners.
        
        This preserves Pareto optimality by using actual ranking differences rather than
        statistical variance. The epsilon for each environment is defined as the absolute
        difference between the 1st and (rank_gap+1)th ranked miner's ELO scores.
        
        Args:
            elo_rankings: ELO scores per hotkey per env
            envs: Environment names
            rank_gap: Index gap for epsilon calculation (default: config.RANK_GAP, typically 2 for 3rd place)
            
        Returns:
            Dict mapping env_name -> epsilon value
        """
        if rank_gap is None:
            rank_gap = self.config.RANK_GAP
            
        epsilon_per_env = {}
        
        for e in envs:
            scores = [elo_rankings[hk][e] for hk in elo_rankings if e in elo_rankings[hk]]
            scores_sorted = sorted(scores, reverse=True)
            
            # If insufficient miners, use floor epsilon
            if len(scores_sorted) <= rank_gap:
                epsilon_per_env[e] = self.config.EPS_FLOOR * 100
            else:
                # Use gap between 1st and (rank_gap+1)th miner
                gap = abs(scores_sorted[0] - scores_sorted[rank_gap])
                epsilon_per_env[e] = max(self.config.EPS_FLOOR * 100, gap)
        
        return epsilon_per_env
    
    
    def calculate_combinatoric_scores(
        self,
        envs: Tuple[str, ...],
        pool: Set[str],
        elo: Dict[str, Dict[str, float]],
        first_block: Dict[str, int],
        epsilon: Dict[str, float],
        scale: float
    ) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Dict[str, str]]:
        """
        Compute signed Nash scores.

        For each env e and miner h:
          delta = elo[h,e] - max_other[e]
          p = max(0, delta - epsilon[e])
          n = max(0, -delta - epsilon[e])
          contrib_e = log1p(p) - gamma * log1p(n)

        The final score is the sum of contrib_e across environments.

        Returns: (scores_by_hk, layer_points_by_hk, env_winners)
          - layer_points exposes a simple per-miner aggregate of positive contributions (layer 1)
          - env_winners marks the top positive-margin miner per env (if any)
        """
        n_envs = len(envs)
        gamma = getattr(self.config, 'NASH_GAMMA', 1.0)
        eps_scale = getattr(self.config, 'NASH_EPS_SCALE', 0.5)
        eps_floor = getattr(self.config, 'NASH_EPS_FLOOR_ELO', 5.0)

        scores: Dict[str, float] = defaultdict(float)
        layer_points: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        env_winners: Dict[str, str] = {}
        debug_info: Dict[str, Tuple[float, int, List[float]]] = {}

        pool_list = list(pool)

        # Determine per-env winners by highest positive (delta - adjusted_epsilon)
        for e in envs:
            best_hk = None
            best_val = 0.0
            adj_eps = max(eps_floor, epsilon.get(e, 0.0) * eps_scale)
            for hk in pool_list:
                if len(pool_list) > 1:
                    max_other = max(elo[g][e] for g in pool_list if g != hk)
                else:
                    max_other = elo[hk][e]
                delta = elo[hk][e] - max_other
                p = max(0.0, delta - adj_eps)
                if p > best_val:
                    best_val = p
                    best_hk = hk
            if best_hk is not None and best_val > 0.0:
                env_winners[e] = best_hk

        # Compute signed Nash score per miner
        for hk in pool_list:
            pos_count = 0
            s_val = 0.0
            pos_sum = 0.0  # for layer 1 display only
            per_env_pos: List[float] = []
            for e in envs:
                if len(pool_list) > 1:
                    max_other = max(elo[g][e] for g in pool_list if g != hk)
                else:
                    max_other = elo[hk][e]
                delta = elo[hk][e] - max_other
                eps = max(eps_floor, epsilon.get(e, 0.0) * eps_scale)
                p = max(0.0, delta - eps)
                n = max(0.0, -delta - eps)
                if p > 0.0:
                    pos_count += 1
                    pos_sum += math.log1p(p)
                per_env_pos.append(p)
                s_val += math.log1p(p) - gamma * math.log1p(n)
            # avoid exact zero to keep a tiny positive signal
            if s_val == 0.0:
                s_val = 0.01
            scores[hk] = s_val
            if pos_sum > 0.0:
                layer_points[hk][1] = pos_sum
            debug_info[hk] = (s_val, pos_count, per_env_pos)

        # Optional debug: print final scores and per-env positive margins
        if os.getenv("AFFINE_NASH_DEBUG", "").lower() in ("1", "true", "yes"):
            try:
                print(f"Nash debug: gamma={gamma}")
                print("Per-env epsilons:", {e: epsilon.get(e, 0.0) for e in envs})
                for hk, (sv, pc, p_list) in debug_info.items():
                    p_str = ", ".join(f"{v:.2f}" for v in p_list)
                    print(f"  hk={hk} score={sv:.4f} pos_envs={pc} p=[{p_str}]")
            except Exception:
                pass

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
        Dict[str, int],
        List[Tuple[Any, Any]]
    ]:
        """
        Process raw sample data into structured counts, metadata, and pairwise results.
        
        Returns:
            Tuple of (cnt, succ, prev, v_id, first_block, pairs_by_time)
        """
        cnt = {hk: defaultdict(int) for hk in meta_hotkeys}
        succ = {hk: defaultdict(int) for hk in meta_hotkeys}
        prev = {}
        v_id = {}
        first_block = {}
        pairs_by_time = []
        
        pending_pairs = {}
        
        for result in results:
            hk = result.miner.hotkey
            env = result.challenge.env
            
            if hk not in cnt:
                continue
                
            try:
                name = result.miner.model.split("/", 1)[1].lower()
            except Exception:
                name = str(result.miner.model).lower()
                
            if hk != base_hk and not name.startswith("affine"):
                continue
            
            cur_vid = (result.miner.model, result.miner.revision)
            
            if v_id.get(hk) != cur_vid:
                v_id[hk] = cur_vid
                first_block[hk] = result.miner.block
                for e in envs:
                    cnt[hk][e] = 0
                    succ[hk][e] = 0
            else:
                try:
                    fb = int(first_block.get(hk, result.miner.block)) if first_block.get(hk) is not None else int(result.miner.block)
                    cb = int(result.miner.block) if result.miner.block is not None else fb
                    first_block[hk] = fb if fb <= cb else cb
                except Exception:
                    pass
            
            prev[hk] = result
            cnt[hk][env] += 1
            succ[hk][env] += float(result.evaluation.score)
            
            # Use challenge_id to match same tasks for ELO pairwise comparison
            challenge_id = result.challenge.challenge_id
            if challenge_id in pending_pairs:
                other_result = pending_pairs.pop(challenge_id)
                pairs_by_time.append((other_result, result))
            else:
                pending_pairs[challenge_id] = result

        return cnt, succ, prev, v_id, first_block, pairs_by_time
    
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
            
        # Normalize weights by the sum of positive scores only
        total_positive = sum(max(0.0, scores.get(hk, 0.0)) for hk in eligible)

        if total_positive > 0.0:
            weight_by_hk = {
                hk: (max(0.0, scores.get(hk, 0.0)) / total_positive) for hk in eligible
            }
        else:
            # Fallback: if no positive scores, pick the best by raw score
            best = max(eligible, key=lambda hk: scores.get(hk, 0.0))
            weight_by_hk = {hk: (1.0 if hk == best else 0.0) for hk in eligible}
        
        # Apply burn if requested
        if burn > 0:
            weight_by_hk, eligible = self.sampler.apply_burn(
                weight_by_hk, burn, base_hotkey, eligible
            )
            
        return weight_by_hk, eligible