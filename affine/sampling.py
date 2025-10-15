"""
Sampling module for Affine validator.
Handles miner sampling and weight calculation using ELO-based epsilon-Pareto dominance.
"""

import math
import itertools
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
            env = result_a.challenge.env.name
            
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
    
    def compute_epsilon_from_elo_variance(
        self,
        elo_rankings: Dict[str, Dict[str, float]],
        envs: Tuple[str, ...]
    ) -> float:
        """
        Calculate epsilon for dominance based on average variance of ELO rankings per env.
        
        Args:
            elo_rankings: ELO scores per hotkey per env
            envs: Environment names
            
        Returns:
            Epsilon value for dominance comparison
        """
        variances = []
        for e in envs:
            scores = [elo_rankings[hk][e] for hk in elo_rankings if e in elo_rankings[hk]]
            if len(scores) > 1:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                variances.append(variance)
        
        if not variances:
            return self.config.EPS_FLOOR * 100
        
        avg_variance = sum(variances) / len(variances)
        return max(self.config.EPS_FLOOR * 100, math.sqrt(avg_variance) / 10.0)
    
    def dominates_on(
        self,
        a: str,
        b: str,
        subset: Tuple[str, ...],
        elo: Dict[str, Dict[str, float]],
        first_block: Dict[str, int],
        epsilon: float
    ) -> bool:
        """
        Check if miner 'a' dominates miner 'b' on the given environment subset using ELO scores.
        Uses epsilon-Pareto dominance with tie-breaking by earliest block.
        
        Args:
            a: Hotkey of miner A
            b: Hotkey of miner B
            subset: Environment subset to compare on
            elo: ELO scores per hotkey per env
            first_block: First block seen for each hotkey
            epsilon: Tolerance for dominance comparison
            
        Returns:
            True if A dominates B
        """
        not_worse_all = True
        better_any = False
        tie_all = True
        
        for e in subset:
            elo_a = elo[a][e]
            elo_b = elo[b][e]
            
            if elo_a < elo_b - epsilon:
                not_worse_all = False
            if elo_a >= elo_b + epsilon:
                better_any = True
            if abs(elo_a - elo_b) > epsilon:
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
        elo: Dict[str, Dict[str, float]],
        first_block: Dict[str, int],
        epsilon: float
    ) -> Dict[str, int]:
        """
        Compute dominance counts for the given miner pool using ELO scores.
        
        Args:
            pool: Set of hotkeys to compare
            envs: Environment names
            elo: ELO scores per hotkey per env
            first_block: First block seen for each hotkey
            epsilon: Tolerance for dominance comparison
            
        Returns:
            Dict mapping hotkey -> dominance count
        """
        dom_counts = defaultdict(int)
        
        for a, b in itertools.permutations(pool, 2):
            if self.dominates_on(a, b, envs, elo, first_block, epsilon):
                dom_counts[a] += 1
                
        return dom_counts
    
    def find_subset_winner(
        self,
        env_subset: Tuple[str, ...],
        pool: Set[str],
        elo: Dict[str, Dict[str, float]],
        first_block: Dict[str, int],
        epsilon: float
    ) -> Optional[str]:
        """
        Find winner on environment subset via ELO-based epsilon-Pareto dominance.
        Falls back to earliest version start block if no dominance edges.
        
        Args:
            env_subset: Environment subset to find winner for
            pool: Set of hotkeys to compare
            elo: ELO scores per hotkey per env
            first_block: First block seen for each hotkey
            epsilon: Tolerance for dominance comparison
            
        Returns:
            Winning hotkey or None
        """
        if not pool:
            return None
            
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool, 2):
            if self.dominates_on(x, y, env_subset, elo, first_block, epsilon):
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
        elo: Dict[str, Dict[str, float]],
        first_block: Dict[str, int],
        epsilon: float,
        scale: float
    ) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Dict[str, str]]:
        """
        Calculate combinatoric scores for all miners using ELO-based dominance.
        
        Args:
            envs: Environment names
            pool: Set of hotkeys to score
            elo: ELO scores per hotkey per env
            first_block: First block seen for each hotkey
            epsilon: Tolerance for dominance comparison
            scale: Scaling factor for layer weights
            
        Returns:
            Tuple of (scores, layer_points, env_winners)
        """
        n_envs = len(envs)
        K = self.compute_layer_weights(n_envs, scale)
        
        scores = defaultdict(float)
        layer_points = defaultdict(lambda: defaultdict(float))
        env_winners = {}
        
        for e in envs:
            env_winners[e] = self.find_subset_winner((e,), pool, elo, first_block, epsilon)
        
        for s in range(1, n_envs + 1):
            for env_subset in itertools.combinations(envs, s):
                winner = self.find_subset_winner(env_subset, pool, elo, first_block, epsilon)
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
            if result.response.success:
                cnt[hk][env] += 1
                succ[hk][env] += float(result.evaluation.score)
            
            challenge_id = str(result.challenge)
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