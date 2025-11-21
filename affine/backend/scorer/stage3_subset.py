"""
Stage 3: Subset Scoring and Weight Distribution

Calculates geometric mean scores for miners within each subset and
distributes weights proportionally based on performance.
"""

import logging
from typing import Dict, List, Any
from models import (
    MinerData,
    SubsetInfo,
    Stage3Output,
)
from config import ScorerConfig
from utils import (
    generate_all_subsets,
    calculate_layer_weights,
    calculate_subset_weights,
    geometric_mean,
    round_score,
)

from affine.core.setup import logger


class Stage3SubsetScorer:
    """Stage 3: Subset Scoring and Weight Distribution.
    
    Responsibilities:
    1. Generate all subset combinations (L1, L2, L3, ...)
    2. Calculate layer weights with exponential growth
    3. For each subset:
       - Calculate geometric mean scores for participating miners
       - Rank miners by score
       - Distribute subset weight proportionally
    4. Apply optional rank-based decay
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize Stage 3 subset scorer.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        self.use_geometric_mean = config.USE_GEOMETRIC_MEAN
        self.apply_rank_decay = config.APPLY_RANK_DECAY
        self.decay_factor = config.DECAY_FACTOR
        self.score_precision = config.SCORE_PRECISION
    
    def score(
        self,
        miners: Dict[int, MinerData],
        environments: List[str]
    ) -> Stage3Output:
        """Calculate subset scores and distribute weights.
        
        Args:
            miners: Dict of MinerData objects from Stage 2
            environments: List of environment names
            
        Returns:
            Stage3Output with subset scores and weights
        """
        logger.info(f"Stage 3: Starting subset scoring for {len(environments)} environments")
        
        # Generate all subsets
        subsets_meta = generate_all_subsets(environments)
        logger.info(f"Generated {len(subsets_meta)} subsets across {len(environments)} layers")
        
        # Calculate layer and subset weights
        layer_weights = calculate_layer_weights(len(environments), self.config.SUBSET_WEIGHT_EXPONENT)
        subset_weights = calculate_subset_weights(subsets_meta, layer_weights)
        
        # Create SubsetInfo objects
        subsets: Dict[str, SubsetInfo] = {}
        for subset_key, subset_meta in subsets_meta.items():
            layer = subset_meta["layer"]
            envs = subset_meta["envs"]
            
            subsets[subset_key] = SubsetInfo(
                key=subset_key,
                layer=layer,
                envs=envs,
                layer_weight=layer_weights[layer],
                subset_weight=subset_weights[subset_key]
            )
        
        # Score each subset
        for subset_key, subset_info in subsets.items():
            self._score_subset(subset_key, subset_info, miners)
        
        # Calculate layer contributions for each miner
        for miner in miners.values():
            layer_totals = {}
            for subset_key, weight in miner.subset_weights.items():
                layer = subsets[subset_key].layer
                layer_totals[layer] = layer_totals.get(layer, 0.0) + weight
            miner.layer_weights = layer_totals
        
        logger.info(f"Stage 3: Completed subset scoring for {len(subsets)} subsets")
        
        return Stage3Output(
            miners=miners,
            subsets=subsets
        )
    
    def _score_subset(
        self,
        subset_key: str,
        subset_info: SubsetInfo,
        miners: Dict[int, MinerData]
    ):
        """Score miners within a single subset and distribute weights.
        
        Args:
            subset_key: Subset identifier
            subset_info: Subset metadata
            miners: Dict of all miners
        """
        envs = subset_info.envs
        
        # Find miners eligible for this subset
        eligible_miners = []
        for miner in miners.values():
            # Skip if filtered from this subset
            if subset_key in miner.filtered_subsets:
                subset_info.filtered_miners.append(miner.uid)
                continue
            
            # Check if miner has valid scores in all subset environments
            has_all_envs = all(
                env in miner.env_scores and miner.env_scores[env].is_valid
                for env in envs
            )
            
            if has_all_envs:
                eligible_miners.append(miner)
                subset_info.valid_miners.append(miner.uid)
        
        # Skip if no eligible miners
        if not eligible_miners:
            return
        
        # Calculate geometric mean scores
        miner_scores = []
        for miner in eligible_miners:
            env_scores = [
                miner.env_scores[env].avg_score
                for env in envs
            ]
            
            if self.use_geometric_mean:
                score = geometric_mean(env_scores)
            else:
                # Fallback to arithmetic mean
                score = sum(env_scores) / len(env_scores)
            
            score = round_score(score, self.score_precision)
            miner_scores.append((miner.uid, score))
            miner.subset_scores[subset_key] = score
        
        # Sort by score (descending)
        miner_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Assign ranks
        for rank, (uid, score) in enumerate(miner_scores, start=1):
            miners[uid].subset_ranks[subset_key] = rank
        
        # Apply rank-based decay if enabled
        if self.apply_rank_decay:
            adjusted_scores = []
            for rank, (uid, score) in enumerate(miner_scores, start=1):
                adjusted = score * (self.decay_factor ** (rank - 1))
                adjusted_scores.append((uid, adjusted))
        else:
            adjusted_scores = miner_scores
        
        # Calculate proportional weights
        total_score = sum(score for _, score in adjusted_scores)
        
        if total_score > 0:
            for uid, score in adjusted_scores:
                proportion = score / total_score
                weight_contribution = subset_info.subset_weight * proportion
                miners[uid].subset_weights[subset_key] = weight_contribution
        else:
            # Edge case: all scores are 0
            equal_weight = subset_info.subset_weight / len(adjusted_scores)
            for uid, _ in adjusted_scores:
                miners[uid].subset_weights[subset_key] = equal_weight
    
    def print_summary(self, output: Stage3Output):
        """Print Stage 3 summary for debugging.
        
        Args:
            output: Stage 3 output data
        """
        logger.info("=" * 80)
        logger.info("STAGE 3 SUMMARY: Subset Scoring")
        logger.info("=" * 80)
        logger.info(f"Total Subsets: {len(output.subsets)}")
        logger.info("")
        
        # Layer statistics
        layer_stats = {}
        for subset in output.subsets.values():
            layer = subset.layer
            if layer not in layer_stats:
                layer_stats[layer] = {
                    'count': 0,
                    'total_weight': 0.0,
                    'avg_miners': []
                }
            layer_stats[layer]['count'] += 1
            layer_stats[layer]['total_weight'] += subset.subset_weight
            layer_stats[layer]['avg_miners'].append(len(subset.valid_miners))
        
        logger.info("Per-Layer Statistics:")
        for layer in sorted(layer_stats.keys()):
            stats = layer_stats[layer]
            avg_miners = sum(stats['avg_miners']) / len(stats['avg_miners']) if stats['avg_miners'] else 0
            logger.info(
                f"  L{layer}: "
                f"Subsets={stats['count']}, "
                f"TotalWeight={stats['total_weight']:.2f}, "
                f"AvgMiners={avg_miners:.1f}"
            )
        
        logger.info("")
        
        # Top miners by cumulative weight
        miner_weights = []
        for uid, miner in output.miners.items():
            cumulative = sum(miner.subset_weights.values())
            miner_weights.append((uid, cumulative, len(miner.subset_weights)))
        
        miner_weights.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Top 10 Miners by Cumulative Weight:")
        for uid, weight, subset_count in miner_weights[:10]:
            hotkey = output.miners[uid].hotkey[:8]
            logger.info(
                f"  UID {uid} ({hotkey}...): "
                f"Weight={weight:.6f}, "
                f"Subsets={subset_count}"
            )
        
        logger.info("=" * 80)