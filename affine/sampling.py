"""
Sampling module for Affine validator.
Handles miner sampling and weight calculation using challenge-based Pareto dominance.
"""

import math
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set


class SamplingConfig:
    """Configuration parameters for sampling."""
    
    TAIL = 50_000  # Increased from 20k to 50k blocks for better statistical stability
    MIN_SAMPLES_PER_ENV = 200  # Increased from 100 to 200 for stronger confidence intervals
    MAX_SAMPLES_CAP = 5000  # Increased from 2000 to 5000
    ELIG = 0.10  # Eligibility threshold: 10% of max samples
    
    # Challenge algorithm parameters
    # Confidence level for Wilson score interval (can be adjusted easily)
    CONFIDENCE_LEVEL = 0.80  # confidence level
    
    @classmethod
    def get_z_score(cls) -> float:
        """
        Calculate Z-score from confidence level.
        Common values:
        - 0.90 → 1.645
        - 0.95 → 1.96
        - 0.99 → 2.576
        """
        from scipy import stats
        return stats.norm.ppf((1 + cls.CONFIDENCE_LEVEL) / 2)
    
    # Environment-specific score ranges for normalization
    # Most environments use 0-1 range, but sciworld uses -100 to 100
    ENV_SCORE_RANGES = {
        "agentgym:sciworld": (-100.0, 100.0),
        # All other environments default to (0.0, 1.0)
    }
    
    @classmethod
    def get_score_range(cls, env: str) -> tuple[float, float]:
        """Get the score range for a specific environment."""
        return cls.ENV_SCORE_RANGES.get(env, (0.0, 1.0))
    
    @classmethod
    def normalize_score(cls, score: float, env: str) -> float:
        """Normalize a score from environment-specific range to [0, 1]."""
        min_score, max_score = cls.get_score_range(env)
        if max_score == min_score:
            return 0.0
        return (score - min_score) / (max_score - min_score)


class ChallengeAlgorithm:
    """
    Bayesian confidence interval based challenge algorithm for miner comparison.
    Uses Wilson score interval to determine winner in single environment.
    """
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
    
    def wilson_score_interval(self, successes: float, trials: int) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.
        
        Args:
            successes: Total score sum (e.g., sum of evaluation scores)
            trials: Number of samples
        
        Returns:
            (lower_bound, upper_bound) confidence interval based on CONFIDENCE_LEVEL
        """
        if trials == 0:
            return (0.0, 0.0)
        
        # Clamp p_hat to [0, 1] to handle edge cases where scores might exceed expected range
        p_hat = min(1.0, max(0.0, successes / trials))
        z = self.config.get_z_score()
        n = trials
        
        denominator = 1 + z**2 / n
        center = p_hat + z**2 / (2 * n)
        
        # Ensure non-negative value under sqrt
        variance_term = p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)
        variance_term = max(0.0, variance_term)
        margin = z * math.sqrt(variance_term)
        
        lower = (center - margin) / denominator
        upper = (center + margin) / denominator
        
        return (max(0.0, lower), min(1.0, upper))
    
    def challenge_winner(
        self,
        stats_a: Dict[str, Any],
        stats_b: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine winner between two miners in a single environment.
        
        Args:
            stats_a: {
                'hotkey': str,
                'samples': int,
                'total_score': float,
                'first_block': int
            }
            stats_b: Same structure as stats_a
        
        Returns:
            'a' | 'b' | None (tie)
        
        Logic:
        1. Miners with insufficient samples (< MIN_SAMPLES_PER_ENV) auto-lose
        2. Calculate confidence intervals and averages for both
        3. If B submitted later, B must satisfy: lower_b > avg_a to win
        4. If A submitted later, A must satisfy: lower_a > avg_b to win
        5. Otherwise, earlier submitter wins (anti-plagiarism)
        """
        samples_a = stats_a['samples']
        samples_b = stats_b['samples']
        min_samples = self.config.MIN_SAMPLES_PER_ENV
        
        # Sample count validation
        if samples_a < min_samples and samples_b < min_samples:
            return None  # Both insufficient
        if samples_a < min_samples:
            return 'b'
        if samples_b < min_samples:
            return 'a'
        
        # Calculate confidence intervals
        lower_a, upper_a = self.wilson_score_interval(
            stats_a['total_score'], samples_a
        )
        lower_b, upper_b = self.wilson_score_interval(
            stats_b['total_score'], samples_b
        )
        
        # Calculate average scores
        avg_a = stats_a['total_score'] / samples_a if samples_a > 0 else 0.0
        avg_b = stats_b['total_score'] / samples_b if samples_b > 0 else 0.0
        
        first_block_a = stats_a['first_block']
        first_block_b = stats_b['first_block']
        
        # Anti-plagiarism: later submitter must outperform average
        if first_block_a < first_block_b:
            # B is later, B needs lower_b > avg_a to win
            if lower_b > avg_a:
                return 'b'
            else:
                return 'a'  # A wins (earlier submission, anti-plagiarism)
        elif first_block_b < first_block_a:
            # A is later, A needs lower_a > avg_b to win
            if lower_a > avg_b:
                return 'a'
            else:
                return 'b'  # B wins (earlier submission, anti-plagiarism)
        else:
            # Same block (rare), compare confidence intervals directly
            if lower_a > upper_b:
                return 'a'
            elif lower_b > upper_a:
                return 'b'
            else:
                return None  # True tie


class MinerSampler:
    """Handles miner sampling and weight calculation with challenge-based scoring."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        self.challenge_algo = ChallengeAlgorithm(config)
    
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
    
    def dominates_on(
        self,
        a: str,
        b: str,
        subset: Tuple[str, ...],
        stats: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> bool:
        """
        Check if miner 'a' dominates miner 'b' on the given environment subset using challenge algorithm.
        
        Args:
            a: Hotkey of miner A
            b: Hotkey of miner B
            subset: Environment subset to compare on
            stats: {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
            
        Returns:
            True if A dominates B on this subset
            
        Dominance logic (Strict):
        - A must win MORE environments than B in the subset
        - Ties do not contribute to dominance
        """
        wins_a = 0
        wins_b = 0
        
        for e in subset:
            stats_a = stats.get(a, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})
            stats_b = stats.get(b, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})
            
            winner = self.challenge_algo.challenge_winner(
                {'hotkey': a, **stats_a},
                {'hotkey': b, **stats_b}
            )
            
            if winner == 'a':
                wins_a += 1
            elif winner == 'b':
                wins_b += 1
            # winner == None is a tie (doesn't count for either side)
        
        # Strict dominance: A must win MORE environments than B
        return wins_a > wins_b
    
    def compute_dominance_counts(
        self,
        pool: Set[str],
        envs: Tuple[str, ...],
        stats: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, int]:
        """
        Compute dominance counts for the given miner pool using challenge algorithm.
        
        Args:
            pool: Set of hotkeys to compare
            envs: Environment names
            stats: {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
            
        Returns:
            Dict mapping hotkey -> dominance count
        """
        dom_counts = defaultdict(int)
        
        for a, b in itertools.permutations(pool, 2):
            if self.dominates_on(a, b, envs, stats):
                dom_counts[a] += 1
                
        return dom_counts
    
    def find_subset_winner(
        self,
        env_subset: Tuple[str, ...],
        pool: Set[str],
        stats: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> List[str]:
        """
        Find winner(s) on environment subset via challenge-based Pareto dominance.
        Returns list of hotkeys that tie for first place (split rewards equally).
        
        Args:
            env_subset: Environment subset to find winner for
            pool: Set of hotkeys to compare
            stats: {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
            
        Returns:
            List of winning hotkeys (may contain multiple winners if tied)
        """
        if not pool:
            return []
            
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool, 2):
            if self.dominates_on(x, y, env_subset, stats):
                dom_local[x] += 1

        # Find maximum dominance count
        max_dom = max((dom_local.get(hk, 0) for hk in pool), default=0)
        
        # Find all miners with maximum dominance count (tied winners)
        tied_winners = [hk for hk in pool if dom_local.get(hk, 0) == max_dom]
        
        return tied_winners
    
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
        stats: Dict[str, Dict[str, Dict[str, Any]]],
        scale: float
    ) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Dict[str, str]]:
        """
        Calculate combinatoric scores for all miners using challenge-based dominance.
        
        Args:
            envs: Environment names
            pool: Set of hotkeys to score
            stats: {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
            scale: Scaling factor for layer weights
            
        Returns:
            Tuple of (scores, layer_points, env_winners)
        """
        n_envs = len(envs)
        K = self.compute_layer_weights(n_envs, scale)
        
        scores = defaultdict(float)
        layer_points = defaultdict(lambda: defaultdict(float))
        env_winners = {}
        
        # Calculate env_winners (pick earliest for display purposes)
        for e in envs:
            winners = self.find_subset_winner((e,), pool, stats)
            if winners:
                # For display, pick the earliest submitter among tied winners
                def get_first_block(hk: str) -> int:
                    return stats.get(hk, {}).get(e, {}).get('first_block', float('inf'))
                env_winners[e] = min(winners, key=get_first_block)
            else:
                env_winners[e] = None

        # Calculate scores with tie-splitting
        for s in range(1, n_envs + 1):
            for env_subset in itertools.combinations(envs, s):
                winners = self.find_subset_winner(env_subset, pool, stats)
                if winners:
                    # Split reward equally among all tied winners
                    reward_per_winner = K[s] / len(winners)
                    for winner in winners:
                        scores[winner] += reward_per_winner
                        layer_points[winner][s] += reward_per_winner
        
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
        Dict[str, Dict[str, Dict[str, Any]]]
    ]:
        """
        Process raw sample data into structured counts and metadata.
        
        Returns:
            Tuple of (cnt, succ, prev, v_id, first_block, stats)
            where stats = {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
        """
        cnt = {hk: defaultdict(int) for hk in meta_hotkeys}
        succ = {hk: defaultdict(int) for hk in meta_hotkeys}
        prev = {}
        v_id = {}
        first_block = {}
        
        # Stats structure for challenge algorithm
        stats = {hk: {} for hk in meta_hotkeys}
        env_first_block = {hk: {e: float('inf') for e in envs} for hk in meta_hotkeys}
        
        # Sort results by timestamp to ensure deterministic pairing
        # This prevents randomness in pending_pairs matching for same challenge_id
        sorted_results = sorted(
            results,
            key=lambda r: r.response.timestamp if r.response.timestamp is not None else 0.0
        )
        for result in sorted_results:
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
                    env_first_block[hk][e] = float('inf')
            else:
                try:
                    fb = int(first_block.get(hk, result.miner.block)) if first_block.get(hk) is not None else int(result.miner.block)
                    cb = int(result.miner.block) if result.miner.block is not None else fb
                    first_block[hk] = fb if fb <= cb else cb
                except Exception:
                    pass
            
            prev[hk] = result
            cnt[hk][env] += 1
            # Normalize score to [0, 1] range based on environment-specific score range
            normalized_score = SamplingConfig.normalize_score(float(result.evaluation.score), env)
            succ[hk][env] += normalized_score
            
            # Track first block per environment
            try:
                block_num = int(result.miner.block)
                if block_num < env_first_block[hk][env]:
                    env_first_block[hk][env] = block_num
            except Exception:
                pass
        
        # Build stats structure for challenge algorithm
        for hk in meta_hotkeys:
            for e in envs:
                stats[hk][e] = {
                    'samples': cnt[hk][e],
                    'total_score': succ[hk][e],
                    'first_block': env_first_block[hk][e] if env_first_block[hk][e] != float('inf') else first_block.get(hk, float('inf'))
                }

        return cnt, succ, prev, v_id, first_block, stats
    
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