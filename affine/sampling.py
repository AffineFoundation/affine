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
    
    TAIL = 20_000  # Ensure all the dataset evaluation result is included in the window
    SCALE = 1.0  # Scaling factor for layer weights
    
    # Task ID deduplication threshold
    # Small datasets (<400) allow 2 samples per task_id, large datasets allow 1
    SMALL_DATASET_THRESHOLD = 400
    
    # Challenge algorithm parameters
    # Confidence level for Beta distribution interval (can be adjusted easily)
    CONFIDENCE_LEVEL = 0.85  # confidence level

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
        confidence_interval_b: Optional[Tuple[float, float]] = None,
        min_samples: int = 0
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
            min_samples: Minimum samples required for this environment (dataset size)

        Returns:
            'a' | 'b' | None (tie)

        Logic:
        1. Miners with insufficient samples (< min_samples) auto-lose
        2. Use pre-computed Beta distribution confidence intervals (or calculate if not provided)
        3. If B submitted later, B must satisfy: lower_b > upper_a to win
        4. If A submitted later, A must satisfy: lower_a > upper_b to win
        5. Otherwise, earlier submitter wins (anti-plagiarism)
        """
        samples_a = stats_a['samples']
        samples_b = stats_b['samples']

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
        # Environment dataset sizes: {env_name: dataset_size}
        # Automatically computed from environment classes
        self.env_dataset_sizes: Dict[str, int] = self._compute_env_dataset_sizes()
    
    @staticmethod
    def _compute_env_dataset_sizes() -> Dict[str, int]:
        """Compute environment dataset sizes from environment class configurations.
        
        Returns:
            {env_name: dataset_size} mapping computed from tasks.py classes
        """
        try:
            from affine.setup import get_enabled_envs
            
            env_dataset_sizes = {}
            for env_class in get_enabled_envs():
                env_name = env_class._env_name
                # Calculate dataset size from class attributes
                start_idx = env_class.DEFAULT_START_INDEX if env_class.DEFAULT_START_INDEX is not None else 0
                end_idx = env_class.DEFAULT_END_INDEX if env_class.DEFAULT_END_INDEX is not None else env_class.DEFAULT_DATA_LEN
                env_dataset_sizes[env_name] = end_idx - start_idx
            
            return env_dataset_sizes
        except ImportError:
            # Fallback if imports fail (e.g., in tests)
            return {}
    
    def calculate_eligibility(
        self,
        cnt: Dict[str, Dict[str, int]],
        active_hks: List[str],
        queryable_hks: Set[str],
        envs: Tuple[str, ...]
    ) -> Tuple[Set[str], Dict[str, int]]:
        """
        Determine eligible miners based on sample requirements.
        
        Each environment requires samples >= dataset_size to be eligible.
        
        Returns:
            Tuple of (eligible_hotkeys, required_samples_per_env)
        """
        required = {}
        
        for e in envs:
            # Minimum samples = dataset size for that environment
            required[e] = SamplingConfig.SMALL_DATASET_THRESHOLD
        
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

        Pareto Dominance:
        - A must be >= B on ALL environments (no environment where B wins)
        - A must be > B on AT LEAST ONE environment (at least one where A wins)
        """
        at_least_one_strict_win = False

        for e in subset:
            stats_a = stats.get(a, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})
            stats_b = stats.get(b, {}).get(e, {'samples': 0, 'total_score': 0.0, 'first_block': float('inf')})

            # Get pre-computed confidence intervals if available
            ci_a = None
            ci_b = None
            if confidence_intervals is not None:
                ci_a = confidence_intervals.get(a, {}).get(e)
                ci_b = confidence_intervals.get(b, {}).get(e)

            # Get minimum samples for this environment
            min_samples = self.env_dataset_sizes.get(e, 0)
            
            winner = self.challenge_algo.challenge_winner(
                {'hotkey': a, **stats_a},
                {'hotkey': b, **stats_b},
                confidence_interval_a=ci_a,
                confidence_interval_b=ci_b,
                min_samples=min_samples
            )

            if winner == 'b':
                # B wins on this environment, so A does NOT dominate B
                return False
            elif winner == 'a':
                # A wins on this environment
                at_least_one_strict_win = True
            # winner == None is a tie (A >= B satisfied, but not strict)

        # A dominates B only if: A >= B on all envs AND A > B on at least one
        return at_least_one_strict_win
    
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
    
    # ========== TEMPORARY FUNCTION - TO BE REMOVED ==========
    # This function filters out task_ids that are outside the valid range
    # for each environment. This is a temporary measure to ensure data consistency
    # during the transition to global sequential sampling.
    # TODO: Remove this function once all data is validated
    @staticmethod
    def _filter_out_of_range_task_ids(
        raw_samples: Dict[str, Dict[str, List[Tuple[float, int, Any]]]],
        envs: Tuple[str, ...],
        env_ranges: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Dict[str, List[Tuple[float, int, Any]]]]:
        """
        [TEMPORARY] Filter out samples with task_ids outside the valid range.
        
        This is a temporary function to ensure task_ids are within the valid
        range for each environment. It should be removed once all miners
        are using the new global sequential sampling.
        
        Args:
            raw_samples: {hotkey: {env: [(score, block_num, task_id), ...]}}
            envs: Tuple of environment names
            env_ranges: {env_name: (start_index, end_index)} valid range for each env
        
        Returns:
            Filtered samples with only valid task_ids
        """
        filtered = {}
        
        for hk, env_samples in raw_samples.items():
            filtered[hk] = {}
            
            for env in envs:
                if env not in env_ranges:
                    # No range specified, keep all samples
                    filtered[hk][env] = env_samples.get(env, [])
                    continue
                
                start_idx, end_idx = env_ranges[env]
                valid_samples = []
                
                for score, block_num, task_id in env_samples.get(env, []):
                    # Check if task_id is within valid range [start_idx, end_idx)
                    if isinstance(task_id, int) and start_idx <= task_id < end_idx:
                        valid_samples.append((score, block_num, task_id))
                    # Skip samples with task_ids outside the range
                
                filtered[hk][env] = valid_samples
        
        return filtered
    # ========== END TEMPORARY FUNCTION ==========
    
    # ========== TEMPORARY HELPER - TO BE REMOVED ==========
    @staticmethod
    def _get_env_ranges(envs: Tuple[str, ...]) -> Dict[str, Tuple[int, int]]:
        """
        [TEMPORARY] Get valid task_id ranges for each environment.
        
        Returns:
            {env_name: (start_index, end_index)} for environments with defined ranges
        """
        try:
            from affine.setup import get_enabled_envs
            
            env_ranges = {}
            for env_class in get_enabled_envs():
                env_name = env_class._env_name
                if env_name in envs:
                    start_idx = env_class.DEFAULT_START_INDEX if env_class.DEFAULT_START_INDEX is not None else 0
                    end_idx = env_class.DEFAULT_END_INDEX if env_class.DEFAULT_END_INDEX is not None else env_class.DEFAULT_DATA_LEN
                    env_ranges[env_name] = (start_idx, end_idx)
            
            return env_ranges
        except ImportError:
            return {}
    # ========== END TEMPORARY HELPER ==========
    
    @staticmethod
    def _deduplicate_samples_by_task_id(
        raw_samples: Dict[str, Dict[str, List[Tuple[float, int, Any]]]],
        envs: Tuple[str, ...],
        env_dataset_sizes: Dict[str, int]
    ) -> Dict[str, Dict[str, List[Tuple[float, int]]]]:
        """
        Deduplicate samples by task_id for each (hotkey, env) combination.
        
        For each unique task_id, keeps a limited number of samples:
        - Large datasets (>= SMALL_DATASET_THRESHOLD): 1 sample per task_id
        - Small datasets (< SMALL_DATASET_THRESHOLD): 2 samples per task_id
        
        This ensures consistent evaluation across all miners by preventing
        multiple evaluations of the same task_id from skewing results.
        
        Args:
            raw_samples: {hotkey: {env: [(score, block_num, task_id), ...]}}
            envs: Tuple of environment names
            env_dataset_sizes: {env_name: dataset_size}
        
        Returns:
            {hotkey: {env: [(score, block_num), ...]}} - deduplicated samples
        """
        # Determine max samples per task_id for each environment
        max_per_task_id = {}
        for e in envs:
            dataset_size = env_dataset_sizes.get(e, 0)
            max_per_task_id[e] = 2 if dataset_size < SamplingConfig.SMALL_DATASET_THRESHOLD else 1
        
        deduplicated = {}
        
        for hk, env_samples in raw_samples.items():
            deduplicated[hk] = {}
            
            for env in envs:
                # Group samples by task_id
                task_id_groups: Dict[Any, List[Tuple[float, int]]] = defaultdict(list)
                
                for score, block_num, task_id in env_samples.get(env, []):
                    samples_for_task = task_id_groups[task_id]
                    
                    if len(samples_for_task) < max_per_task_id.get(env, 1):
                        # Still room for more samples
                        samples_for_task.append((score, block_num))
                    else:
                        # Replace oldest sample if this one is newer
                        oldest_idx = min(range(len(samples_for_task)), key=lambda i: samples_for_task[i][1])
                        if block_num > samples_for_task[oldest_idx][1]:
                            samples_for_task[oldest_idx] = (score, block_num)
                
                # Flatten all task_id groups into a single list
                all_samples = []
                for task_samples in task_id_groups.values():
                    all_samples.extend(task_samples)
                
                deduplicated[hk][env] = all_samples
        
        return deduplicated
    
    def process_sample_data(
        self,
        results: List[Any],
        meta_hotkeys: List[str],
        envs: Tuple[str, ...],
        base_hk: str,
        env_dataset_sizes: Optional[Dict[str, int]] = None
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
        
        Implements task_id deduplication to ensure consistent evaluation:
        - For each (hotkey, env, task_id), keep only unique samples
        - Large datasets (>=400): keep 1 sample per task_id
        - Small datasets (<400): keep up to 2 samples per task_id
        
        Args:
            results: List of evaluation results
            meta_hotkeys: All hotkeys in metagraph
            envs: Environment names
            base_hk: Base hotkey (validator)
            env_dataset_sizes: {env_name: dataset_size} for determining dedup limit
        
        Returns:
            Tuple of (cnt, succ, prev, v_id, first_block, stats)
            where stats = {hotkey: {env: {'samples': int, 'total_score': float, 'first_block': int}}}
        """
        if env_dataset_sizes is None:
            env_dataset_sizes = {}
        
        cnt = {hk: defaultdict(int) for hk in meta_hotkeys}
        succ = {hk: defaultdict(int) for hk in meta_hotkeys}
        prev = {}
        v_id = {}
        first_block = {}

        # Stats structure for challenge algorithm
        stats = {hk: {} for hk in meta_hotkeys}
        env_first_block = {hk: {e: float('inf') for e in envs} for hk in meta_hotkeys}

        # Collect raw samples with task_id information
        # Structure: {hk: {env: [(score, block_num, task_id), ...]}}
        raw_samples = {hk: {e: [] for e in envs} for hk in meta_hotkeys}

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
                    # Clear samples when version changes
                    raw_samples[hk][e].clear()
            else:
                try:
                    fb = int(first_block.get(hk, result.miner.block)) if first_block.get(hk) is not None else int(result.miner.block)
                    cb = int(result.miner.block) if result.miner.block is not None else fb
                    first_block[hk] = fb if fb <= cb else cb
                except Exception:
                    pass

            prev[hk] = result

            raw_score = float(result.evaluation.score)

            # Filter out invalid alfworld scores (should be 0 or 1, not other values)
            if "alfworld" in env and raw_score not in [0.0, 1.0]:
                continue

            normalized_score = SamplingConfig.normalize_score(raw_score, env)
            try:
                block_num = int(result.miner.block)
            except Exception:
                block_num = float('inf')
            
            # Get task_id from result (may be None for legacy data)
            task_id = getattr(result, 'task_id', None)
            if task_id is None:
                # For legacy data without task_id, use block_num as unique identifier
                task_id = block_num
            
            # Add raw sample with task_id for later deduplication
            raw_samples[hk][env].append((normalized_score, block_num, task_id))

        # ========== TEMPORARY: Filter out-of-range task_ids ==========
        # TODO: Remove this block once all data is validated
        # Get environment ranges from env classes
        env_ranges = self._get_env_ranges(envs)
        if env_ranges:
            raw_samples = self._filter_out_of_range_task_ids(raw_samples, envs, env_ranges)
        # ========== END TEMPORARY BLOCK ==========

        # Apply task_id deduplication using dedicated function
        deduplicated_samples = self._deduplicate_samples_by_task_id(
            raw_samples, envs, env_dataset_sizes
        )
        
        # Extract final counts and scores from deduplicated samples
        for hk in meta_hotkeys:
            for e in envs:
                all_samples = deduplicated_samples.get(hk, {}).get(e, [])
                
                cnt[hk][e] = len(all_samples)
                succ[hk][e] = sum(score for score, _ in all_samples)
                if all_samples:
                    env_first_block[hk][e] = min(block for _, block in all_samples)
                else:
                    env_first_block[hk][e] = float('inf')
        
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

        # Reassign weights below threshold (0.01) to base_hotkey to avoid low-value tail models
        weight_threshold = 0.01
        reassigned_weight = 0.0

        for hk in list(weight_by_hk.keys()):
            if hk != base_hotkey and weight_by_hk[hk] < weight_threshold:
                reassigned_weight += weight_by_hk[hk]
                weight_by_hk[hk] = 0.0

        if reassigned_weight > 0:
            if base_hotkey in weight_by_hk:
                weight_by_hk[base_hotkey] += reassigned_weight
            else:
                weight_by_hk[base_hotkey] = reassigned_weight

        # Apply burn if requested
        if burn > 0:
            weight_by_hk, eligible = self.sampler.apply_burn(
                weight_by_hk, burn, base_hotkey, eligible
            )

        return weight_by_hk, eligible
