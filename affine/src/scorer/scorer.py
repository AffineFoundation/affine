"""
Main Scorer Orchestrator

Coordinates the four-stage scoring algorithm and manages result persistence.
"""

import time
import logging
from typing import Dict, Any, Optional
from .config import ScorerConfig
from .models import ScoringResult
from .stage1_collector import Stage1Collector
from .stage2_pareto import Stage2ParetoFilter
from .stage3_subset import Stage3SubsetScorer
from .stage4_weights import Stage4WeightNormalizer
from .utils import generate_all_subsets

from affine.core.setup import logger


class Scorer:
    """Main scorer orchestrator.
    
    Coordinates the four-stage scoring algorithm:
    1. Data Collection: Collect and validate sample data
    2. Pareto Filtering: Apply anti-plagiarism filtering
    3. Subset Scoring: Calculate geometric mean scores and distribute weights
    4. Weight Normalization: Apply threshold, burning, and normalization
    
    Optionally saves results to database.
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize scorer with configuration.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        
        # Initialize stage processors
        self.stage1 = Stage1Collector(config)
        self.stage2 = Stage2ParetoFilter(config)
        self.stage3 = Stage3SubsetScorer(config)
        self.stage4 = Stage4WeightNormalizer(config)
    
    def calculate_scores(
        self,
        scoring_data: Dict[str, Any],
        environments: list,
        block_number: int,
        print_summary: bool = True
    ) -> ScoringResult:
        """Execute the four-stage scoring algorithm.
        
        Args:
            scoring_data: Response from /api/v1/samples/scoring
            environments: List of environment names participating in scoring
            block_number: Current block number
            print_summary: Whether to print detailed summaries (default: True)
            
        Returns:
            ScoringResult with complete scoring data
        """
        start_time = time.time()
        logger.info(f"Total Miners: {len(scoring_data)}")
        
        # Stage 1: Data Collection
        stage1_output = self.stage1.collect(scoring_data, environments)
        
        # Stage 2: Pareto Filtering
        # Apply MAX_LAYERS limit: only evaluate top layers (e.g., L3-L8 if 8 envs and MAX_LAYERS=6)
        subsets_meta = generate_all_subsets(environments, max_layers=self.config.MAX_LAYERS)
        stage2_output = self.stage2.filter(stage1_output.miners, subsets_meta)
        
        # Stage 3: Subset Scoring
        stage3_output = self.stage3.score(stage2_output.miners, environments)
        
        # Stage 4: Weight Normalization
        stage4_output = self.stage4.normalize(stage3_output.miners)
        
        # Build final result
        result = ScoringResult(
            block_number=block_number,
            calculated_at=int(time.time()),
            environments=environments,
            config=self.config.to_dict(),
            miners=stage3_output.miners,
            pareto_comparisons=stage2_output.comparisons,
            subsets=stage3_output.subsets,
            final_weights=stage4_output.final_weights,
            total_miners=len(scoring_data),
            valid_miners=stage1_output.valid_count,
            invalid_miners=stage1_output.invalid_count,
            burn_weight=stage4_output.burn_weight
        )
        
        elapsed_time = time.time() - start_time
        non_zero = len([w for w in result.final_weights.values() if w > 0])
        logger.info("=" * 80)
        logger.info(f"SCORING COMPLETED - Time: {elapsed_time:.2f}s, Active: {non_zero}/{len(scoring_data)}")
        logger.info("=" * 80)
        
        # Print detailed summary table
        if print_summary:
            self.stage4.print_detailed_table(stage3_output.miners, environments)
        
        return result
    
    async def save_results(
        self,
        result: ScoringResult,
        miner_scores_dao=None,
        score_snapshots_dao=None,
        scores_dao=None
    ):
        """Save scoring results to database.
        
        Args:
            result: ScoringResult to save
            miner_scores_dao: MinerScoresDAO instance (optional)
            score_snapshots_dao: ScoreSnapshotsDAO instance (optional)
            scores_dao: ScoresDAO instance (optional)
        """
        if not miner_scores_dao or not score_snapshots_dao or not scores_dao:
            logger.warning("DAO instances not provided, skipping database save")
            return
        
        logger.info(f"Saving scoring results to database (block {result.block_number})")
        
        # Save individual miner scores to miner_scores table (detailed)
        for uid, miner in result.miners.items():
            # Prepare environment scores
            env_scores = {
                env: {
                    "avg_score": score.avg_score,
                    "sample_count": score.sample_count,
                    "completeness": score.completeness
                }
                for env, score in miner.env_scores.items()
            }
            
            # Prepare layer scores
            layer_scores = {
                f"L{layer}": weight
                for layer, weight in miner.layer_weights.items()
            }
            
            # Prepare subset contributions
            subset_contributions = {
                subset_key: {
                    "score": miner.subset_scores.get(subset_key, 0.0),
                    "rank": miner.subset_ranks.get(subset_key, 0),
                    "weight": weight
                }
                for subset_key, weight in miner.subset_weights.items()
            }
            
            # Prepare filter info
            filter_info = {
                "filtered_subsets": miner.filtered_subsets,
                "filter_reasons": miner.filter_reasons
            }
            
            # Save miner score
            await miner_scores_dao.save_miner_score(
                block_number=result.block_number,
                hotkey=miner.hotkey,
                uid=uid,
                model_revision=miner.model_revision,
                env_scores=env_scores,
                layer_scores=layer_scores,
                subset_contributions=subset_contributions,
                cumulative_weight=miner.cumulative_weight,
                normalized_weight=miner.normalized_weight,
                filter_info=filter_info
            )
        
        # Save snapshot metadata
        statistics = {
            "total_miners": result.total_miners,
            "valid_miners": result.valid_miners,
            "invalid_miners": result.invalid_miners,
            "miner_final_scores": {
                str(uid): weight
                for uid, weight in result.final_weights.items()
            }
        }
        
        await score_snapshots_dao.save_snapshot(
            block_number=result.block_number,
            scorer_hotkey="scorer_service",
            config=result.config,
            statistics=statistics
        )
        
        # Save to scores table (simplified, for API queries)
        logger.info(f"Saving to scores table for API queries...")
        for uid, miner in result.miners.items():
            # Calculate total samples
            total_samples = sum(
                env_score.sample_count
                for env_score in miner.env_scores.values()
            )
            
            # Prepare scores by environment
            scores_by_env = {
                env: score.avg_score
                for env, score in miner.env_scores.items()
            }
            
            # Use normalized weight as overall score
            overall_score = miner.normalized_weight
            
            # Simple CI estimation (±10% of score)
            ci_lower = max(0.0, overall_score * 0.9)
            ci_upper = min(1.0, overall_score * 1.1)
            
            # Calculate average score from environments
            if scores_by_env:
                average_score = sum(scores_by_env.values()) / len(scores_by_env)
            else:
                average_score = 0.0
            
            # Convert layer_weights keys from int to str for DynamoDB
            scores_by_layer = {
                f"L{layer}": weight
                for layer, weight in miner.layer_weights.items()
            }
            
            await scores_dao.save_score(
                block_number=result.block_number,
                miner_hotkey=miner.hotkey,
                uid=uid,
                model_revision=miner.model_revision,
                overall_score=overall_score,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                average_score=average_score,
                scores_by_layer=scores_by_layer,
                scores_by_env=scores_by_env,
                total_samples=total_samples,
                is_eligible=(overall_score > 0),
                meets_criteria=(overall_score > 0)
            )
        
        logger.info(f"Successfully saved scoring results for {len(result.miners)} miners to all tables")


def create_scorer(config: Optional[ScorerConfig] = None) -> Scorer:
    """Factory function to create a Scorer instance.
    
    Args:
        config: Optional custom configuration
        
    Returns:
        Configured Scorer instance
    """
    if config is None:
        config = ScorerConfig()
    
    return Scorer(config)