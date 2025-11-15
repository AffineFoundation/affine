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
    MIN_SAMPLES_PER_ENV = 400  # Increased from 200 to 400 for stronger confidence intervals
    MAX_SAMPLES_CAP = 5000  # Increased from 2000 to 5000
    ELIG = 0.10  # Eligibility threshold: 10% of max samples
    SCALE = 1.0  # Scaling factor for layer weights
    
    # Challenge algorithm parameters
    # Confidence level for Beta distribution interval (can be adjusted easily)
    CONFIDENCE_LEVEL = 0.9  # confidence level

    # Beta distribution prior parameters (Jeffrey's prior for binomial proportion)
    BETA_PRIOR_ALPHA = 0.5
    BETA_PRIOR_BETA = 0.5
    
    # Winner-Takes-More distribution parameters
    # SCORE_POWER: Exponent applied to comprehensive scores before distribution
    #              Higher values (e.g., 2.0) amplify differences between top performers
    # RANK_DECAY_RATE: Decay factor for subsequent ranks (0.0-1.0)
    #                  0.0 = winner takes all, 1.0 = no decay (proportional to scores)
    SCORE_POWER = 2.0
    RANK_DECAY_RATE = 0.5
    
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
    Uses Beta distribution with Jeffrey's prior to determine winner in single environment.
    """
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
    
    def beta_confidence_interval(self, successes: float, trials: int) -> Tuple[float, float]:
        """
        Calculate Beta distribution confidence interval using Bayesian approach.

        Uses Jeffrey's prior Beta(0.5, 0.5) which is uninformative and performs well
        for small samples and extreme proportions.

        Args:
            successes: Total score sum (e.g., sum of evaluation scores)
            trials: Number of samples

        Returns:
            (lower_bound, upper_bound) confidence interval based on CONFIDENCE_LEVEL
        """
        if trials == 0:
            return (0.0, 0.0)

        from scipy.stats import beta

        # Clamp successes to valid range
        successes = min(float(trials), max(0.0, successes))

        # Posterior parameters with Jeffrey's prior
        alpha_posterior = self.config.BETA_PRIOR_ALPHA + successes
        beta_posterior = self.config.BETA_PRIOR_BETA + (trials - successes)

        # Calculate confidence interval using quantiles
        alpha_level = (1 - self.config.CONFIDENCE_LEVEL) / 2
        lower = beta.ppf(alpha_level, alpha_posterior, beta_posterior)
        upper = beta.ppf(1 - alpha_level, alpha_posterior, beta_posterior)

        # Handle edge cases where ppf might return nan
        lower = 0.0 if math.isnan(lower) else max(0.0, lower)
        upper = 1.0 if math.isnan(upper) else min(1.0, upper)

        return (lower, upper)
    
    def challenge_winner(
        self,
        stats_a: Dict[str, Any],
        stats_b: Dict[str, Any],
        confidence_interval_a: Optional[Tuple[float, float]] = None,
        confidence_interval_b: Optional[Tuple[float, float]] = None
    ) -> Optional[str]:
        """
        Determine winner between two miners in a single environment using Beta distribution.

        Args:
            stats_a: {
                'hotkey': str,
                'samples': int,
                'total_score': float,
                'first_block': int
            }
            stats_b: Same structure as stats_a
            confidence_interval_a: Pre-computed confidence interval for A (lower, upper), optional
            confidence_interval_b: Pre-computed confidence interval for B (lower, upper), optional

        Returns:
            'a' | 'b' | None (tie)

        Logic:
        1. Miners with insufficient samples (< MIN_SAMPLES_PER_ENV) auto-lose
        2. Use pre-computed Beta distribution confidence intervals (or calculate if not provided)
        3. If B submitted later, B must satisfy: lower_b > upper_a to win
        4. If A submitted later, A must satisfy: lower_a > upper_b to win
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

        # Use pre-computed confidence intervals or calculate on-demand
        if confidence_interval_a is not None:
            lower_a, upper_a = confidence_interval_a
        else:
            lower_a, upper_a = self.beta_confidence_interval(
                stats_a['total_score'], samples_a
            )

        if confidence_interval_b is not None:
            lower_b, upper_b = confidence_interval_b
        else:
            lower_b, upper_b = self.beta_confidence_interval(
                stats_b['total_score'], samples_b
            )
        
        first_block_a = stats_a['first_block']
        first_block_b = stats_b['first_block']
        
        # Anti-plagiarism: later submitter must significantly outperform
        if first_block_a < first_block_b:
            # B is later, B needs lower_b > upper_a to win
            if lower_b > upper_a:
                return 'b'
            else:
                return 'a'  # A wins (earlier submission, anti-plagiarism)
        elif first_block_b < first_block_a:
            # A is later, A needs lower_a > upper_b to win
            if lower_a > upper_b:
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
        stats: Dict[str, Dict[str, Dict[str, Any]]],
        confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    ) -> bool:
        """
        Check if miner 'a' Pareto-dominates miner 'b' on the given environment subset.

        Args:
            a: Hotkey of miner A
            b: Hotkey of miner B
            subset: Environment subset to compare on
            stats: {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
            confidence_intervals: Pre-computed {hotkey: {env: (lower, upper)}}, optional

        Returns:
            True if A Pareto-dominates B on this subset

        Pareto Dominance (with half-environment threshold):
        - Determine who submitted first (old model) vs later (new model)
        - New model must win on MORE THAN HALF of the total environments to dominate old model
        - Old model can dominate new model using standard Pareto rules (no loss + at least one win)
        - If both submitted at same time, use standard Pareto dominance
        """
        # Determine submission order based on first_block across all environments in subset
        first_block_a = float('inf')
        first_block_b = float('inf')

        for e in subset:
            stats_a = stats.get(a, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})
            stats_b = stats.get(b, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})
            first_block_a = min(first_block_a, stats_a.get('first_block', float('inf')))
            first_block_b = min(first_block_b, stats_b.get('first_block', float('inf')))

        # Count wins for each miner
        a_wins = 0
        b_wins = 0
        ties = 0

        for e in subset:
            stats_a = stats.get(a, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})
            stats_b = stats.get(b, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})

            # Get pre-computed confidence intervals if available
            ci_a = None
            ci_b = None
            if confidence_intervals is not None:
                ci_a = confidence_intervals.get(a, {}).get(e)
                ci_b = confidence_intervals.get(b, {}).get(e)

            winner = self.challenge_algo.challenge_winner(
                {'hotkey': a, **stats_a},
                {'hotkey': b, **stats_b},
                confidence_interval_a=ci_a,
                confidence_interval_b=ci_b
            )

            if winner == 'a':
                a_wins += 1
            elif winner == 'b':
                b_wins += 1
            else:
                ties += 1

        total_envs = len(subset)
        half_threshold = total_envs / 2.0

        # Determine who is "old" and who is "new" based on first_block
        if first_block_a < first_block_b:
            # A is old model, B is new model
            # For A (old) to dominate B (new): standard Pareto (no losses + at least one win)
            if b_wins == 0 and a_wins > 0:
                return True
            else:
                return False
        elif first_block_b < first_block_a:
            # B is old model, A is new model
            # For A (new) to dominate B (old): A must win MORE THAN HALF of environments
            if a_wins > half_threshold:
                return True
            else:
                return False
        else:
            # Same submission time (or both inf), use standard Pareto dominance
            # A dominates B if: no environment where B wins AND at least one where A wins
            return b_wins == 0 and a_wins > 0
    
    def calculate_comprehensive_score(
        self,
        hotkey: str,
        env_subset: Tuple[str, ...],
        stats: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> float:
        """
        Calculate comprehensive ability score for a miner on given environment subset.
        
        Uses geometric mean to naturally penalize specialization while rewarding
        balanced performance across all environments.
        
        Args:
            hotkey: Miner's hotkey
            env_subset: Environment subset to evaluate on
            stats: {hotkey: {env: {'samples': int, 'total_score': float, ...}}}
        
        Returns:
            Comprehensive ability score (geometric mean in [0, 1])
        """
        scores = []
        
        for env in env_subset:
            env_stats = stats.get(hotkey, {}).get(env, {})
            samples = env_stats.get('samples', 0)
            total_score = env_stats.get('total_score', 0.0)
            
            # Calculate normalized score (average performance in [0, 1])
            if samples > 0:
                normalized_score = total_score / samples
                # Clamp to valid range as a safety measure
                normalized_score = max(0.0, min(1.0, normalized_score))
            else:
                normalized_score = 0.0
            
            scores.append(normalized_score)
        
        # Return 0 if any score is zero or negative (geometric mean would be 0)
        if not scores or any(s <= 0.0 for s in scores):
            return 0.0
        
        # Calculate geometric mean as comprehensive score
        geometric_mean = math.prod(scores) ** (1.0 / len(scores))
        
        return geometric_mean
    
    def calculate_decayed_scores(
        self,
        comprehensive_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply score-based weighting with rank decay to comprehensive scores.
        
        Algorithm:
        1. Rank miners by comprehensive score (descending)
        2. Apply power function to amplify score differences: score^SCORE_POWER
        3. Apply rank-based decay: rank_n gets RANK_DECAY_RATE^(n-1) multiplier
        4. Final weight = (score^power) * (decay^rank)
        
        This ensures:
        - Large performance gaps lead to large weight gaps (via SCORE_POWER)
        - Later ranks get progressively less weight (via RANK_DECAY_RATE)
        - Distribution automatically reflects actual performance differences
        
        Args:
            comprehensive_scores: {hotkey: geometric_mean_score}
        
        Returns:
            {hotkey: decayed_score} (not normalized, for proportional distribution)
        """
        if not comprehensive_scores:
            return {}
        
        # Sort by comprehensive score (descending)
        sorted_miners = sorted(
            comprehensive_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate decayed scores
        decayed_scores = {}
        for rank, (hk, comp_score) in enumerate(sorted_miners):
            # Apply power function to amplify differences
            powered_score = comp_score ** self.config.SCORE_POWER
            
            # Apply rank-based decay (rank 0 gets 1.0, rank 1 gets DECAY, etc.)
            rank_multiplier = self.config.RANK_DECAY_RATE ** rank
            
            # Combine both factors
            decayed_scores[hk] = powered_score * rank_multiplier
        
        return decayed_scores
    
    def find_subset_winner(
        self,
        env_subset: Tuple[str, ...],
        pool: Set[str],
        stats: Dict[str, Dict[str, Dict[str, Any]]],
        confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    ) -> Dict[str, float]:
        """
        Find winner(s) on environment subset via Pareto dominance,
        then distribute rewards based on score-weighted rank decay.

        Args:
            env_subset: Environment subset to find winner for
            pool: Set of hotkeys to compare
            stats: {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
            confidence_intervals: Pre-computed {hotkey: {env: (lower, upper)}}, optional

        Returns:
            Dict mapping hotkey -> proportional weight (sums to 1.0)
            Empty dict if no miners in pool
        """
        if not pool:
            return {}

        # Step 1: Find Pareto frontier (miners not dominated by anyone)
        pareto_frontier = []

        for candidate in pool:
            is_dominated = False
            for other in pool:
                if candidate == other:
                    continue
                # Check if 'other' dominates 'candidate'
                if self.dominates_on(other, candidate, env_subset, stats, confidence_intervals):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_frontier.append(candidate)

        # Fallback: if no one is on frontier, include everyone
        if not pareto_frontier:
            pareto_frontier = list(pool)
        
        # Step 2: Single winner takes all
        if len(pareto_frontier) == 1:
            return {pareto_frontier[0]: 1.0}
        
        # Step 3: Calculate comprehensive scores for all Pareto frontier members
        comprehensive_scores = {}
        for hk in pareto_frontier:
            comprehensive_scores[hk] = self.calculate_comprehensive_score(
                hk, env_subset, stats
            )
        
        total_score = sum(comprehensive_scores.values())
        
        # Fallback to equal split if all scores are zero
        if total_score <= 0.0:
            equal_weight = 1.0 / len(pareto_frontier)
            return {hk: equal_weight for hk in pareto_frontier}
        
        # Step 4: Apply score-based weighting with rank decay
        decayed_scores = self.calculate_decayed_scores(comprehensive_scores)
        
        # Step 5: Normalize to sum to 1.0
        total_decayed = sum(decayed_scores.values())
        if total_decayed <= 0.0:
            # Fallback to equal distribution
            equal_weight = 1.0 / len(pareto_frontier)
            return {hk: equal_weight for hk in pareto_frontier}
        
        return {hk: score / total_decayed for hk, score in decayed_scores.items()}
    
    def compute_layer_weights(self, n_envs: int) -> Dict[int, float]:
        """
        Compute per-subset weights K_s.
        
        Only evaluate the top 6 layers to focus on comprehensive performance.
        Each layer gets exponentially increasing total weight: layer_s_total = scale * (2^s)
        This total weight is then evenly distributed among all subsets in that layer.
        
        For layer s with C(n_envs, s) subsets, each subset gets:
        K_s = (scale * 2^s) / C(n_envs, s)
        
        This ensures:
        1. Higher layers (more environments) get exponentially more total weight
        2. Within each layer, weight is fairly distributed across all subsets
        3. Models are incentivized to perform well across more environments
        4. Low layers (L1, L2, etc.) are excluded as they contribute negligibly under exponential growth
        """
        K = {}
        # Only evaluate top 6 layers (dynamically: max(1, n_envs - 5) to n_envs)
        min_layer = max(1, n_envs - 5)
        
        for s in range(min_layer, n_envs + 1):
            # Calculate number of subsets in this layer: C(n_envs, s)
            num_subsets = math.comb(n_envs, s)
            
            # Total weight for this layer (exponential by layer size)
            layer_total_weight = self.config.SCALE * (2 ** s)
            
            # Distribute evenly among all subsets in this layer
            K[s] = layer_total_weight / num_subsets if num_subsets > 0 else layer_total_weight
        
        return K

    def calculate_combinatoric_scores(
        self,
        envs: Tuple[str, ...],
        pool: Set[str],
        stats: Dict[str, Dict[str, Dict[str, Any]]],
        confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    ) -> Tuple[Dict[str, float], Dict[str, Dict[int, float]], Dict[str, str]]:
        """
        Calculate combinatoric scores for all miners using challenge-based dominance
        with comprehensive ability-based reward distribution.

        Args:
            envs: Environment names
            pool: Set of hotkeys to score
            stats: {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
            confidence_intervals: Pre-computed {hotkey: {env: (lower, upper)}}, optional

        Returns:
            Tuple of (scores, layer_points, env_winners)
        """
        n_envs = len(envs)
        K = self.compute_layer_weights(n_envs)

        scores = defaultdict(float)
        layer_points = defaultdict(lambda: defaultdict(float))
        env_winners = {}

        # Calculate env_winners (pick earliest for display purposes)
        for e in envs:
            winner_weights = self.find_subset_winner((e,), pool, stats, confidence_intervals)
            if winner_weights:
                # For display, pick the winner with highest weight (or earliest if tied)
                def sort_key(hk: str) -> Tuple[float, int]:
                    weight = winner_weights.get(hk, 0.0)
                    first_block = stats.get(hk, {}).get(e, {}).get('first_block', float('inf'))
                    return (-weight, first_block)  # Negative weight for descending order
                env_winners[e] = min(winner_weights.keys(), key=sort_key)
            else:
                env_winners[e] = None

        # Calculate scores with proportional distribution based on comprehensive ability
        # Only evaluate top 6 layers (same range as compute_layer_weights)
        min_layer = max(1, n_envs - 5)
        for s in range(min_layer, n_envs + 1):
            for env_subset in itertools.combinations(envs, s):
                # Get proportional weights for winners on this subset
                winner_weights = self.find_subset_winner(env_subset, pool, stats, confidence_intervals)
                
                if winner_weights:
                    # Distribute base reward proportionally according to comprehensive ability
                    base_reward = K[s]
                    for hotkey, weight in winner_weights.items():
                        reward = base_reward * weight
                        scores[hotkey] += reward
                        layer_points[hotkey][s] += reward

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
