"""
Stage 4: Weight Normalization and Finalization

Aggregates subset contributions, applies minimum threshold,
implements burning mechanism, and normalizes final weights.
"""

import logging
from typing import Dict
from models import (
    MinerData,
    Stage4Output,
)
from config import ScorerConfig
from utils import (
    normalize_weights,
    apply_min_threshold,
    apply_burn_mechanism,
)

from affine.core.setup import logger


class Stage4WeightNormalizer:
    """Stage 4: Weight Normalization and Finalization.
    
    Responsibilities:
    1. Accumulate subset weight contributions for each miner
    2. Apply minimum weight threshold (remove miners < 1%)
    3. Apply burning mechanism (allocate percentage to UID 0)
    4. Normalize weights to sum to 1.0
    5. Generate final weight distribution for chain
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize Stage 4 weight normalizer.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        self.min_threshold = config.MIN_WEIGHT_THRESHOLD
        self.burn_percentage = config.BURN_PERCENTAGE
    
    def normalize(
        self,
        miners: Dict[int, MinerData]
    ) -> Stage4Output:
        """Normalize weights and finalize distribution.
        
        Args:
            miners: Dict of MinerData objects from Stage 3
            
        Returns:
            Stage4Output with final normalized weights
        """
        logger.info(f"Stage 4: Starting weight normalization for {len(miners)} miners")
        
        # Step 1: Accumulate cumulative weights
        raw_weights: Dict[int, float] = {}
        for uid, miner in miners.items():
            cumulative = sum(miner.subset_weights.values())
            miner.cumulative_weight = cumulative
            raw_weights[uid] = cumulative
        
        logger.info(f"Accumulated cumulative weights from subset contributions")
        
        # Step 2: Apply minimum threshold
        weights_after_threshold = apply_min_threshold(
            raw_weights,
            self.min_threshold
        )
        
        below_threshold_count = sum(
            1 for uid, weight in raw_weights.items()
            if weight > 0 and weight < self.min_threshold
        )
        
        if below_threshold_count > 0:
            logger.info(
                f"Removed {below_threshold_count} miners below threshold "
                f"({self.min_threshold:.1%})"
            )
        
        # Step 3: Normalize weights (before burning)
        normalized_weights = normalize_weights(weights_after_threshold)
        
        # Step 4: Apply burning mechanism
        burn_weight = 0.0
        if self.burn_percentage > 0:
            normalized_weights, burn_weight = apply_burn_mechanism(
                normalized_weights,
                self.burn_percentage,
                burn_uid=0
            )
            logger.info(
                f"Applied burning: {self.burn_percentage:.1%} → "
                f"{burn_weight:.6f} allocated to UID 0"
            )
        
        # Step 5: Final normalization (ensure sum = 1.0)
        final_weights = normalize_weights(normalized_weights)
        
        # Update miner objects with normalized weights
        for uid, weight in final_weights.items():
            if uid in miners:
                miners[uid].normalized_weight = weight
        
        logger.info(
            f"Stage 4: Completed normalization - "
            f"{len([w for w in final_weights.values() if w > 0])} miners with non-zero weights"
        )
        
        return Stage4Output(
            final_weights=final_weights,
            burn_weight=burn_weight,
            below_threshold_count=below_threshold_count
        )
    
    def print_summary(self, output: Stage4Output, miners: Dict[int, MinerData]):
        """Print Stage 4 summary for debugging.
        
        Args:
            output: Stage 4 output data
            miners: Dict of all miners
        """
        logger.info("=" * 80)
        logger.info("STAGE 4 SUMMARY: Weight Normalization")
        logger.info("=" * 80)
        logger.info(f"Total Miners: {len(miners)}")
        logger.info(f"Non-Zero Weights: {len([w for w in output.final_weights.values() if w > 0])}")
        logger.info(f"Below Threshold: {output.below_threshold_count}")
        logger.info(f"Burn Weight: {output.burn_weight:.6f} ({self.burn_percentage:.1%})")
        logger.info("")
        
        # Weight distribution statistics
        non_zero_weights = [w for w in output.final_weights.values() if w > 0]
        if non_zero_weights:
            logger.info("Weight Distribution:")
            logger.info(f"  Max Weight: {max(non_zero_weights):.6f}")
            logger.info(f"  Min Weight: {min(non_zero_weights):.6f}")
            logger.info(f"  Avg Weight: {sum(non_zero_weights) / len(non_zero_weights):.6f}")
            logger.info(f"  Total Weight: {sum(output.final_weights.values()):.6f}")
        
        logger.info("")
        
        # Top 10 miners by final weight
        sorted_weights = sorted(
            output.final_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info("Top 10 Miners by Final Weight:")
        for uid, weight in sorted_weights[:10]:
            if uid in miners:
                hotkey = miners[uid].hotkey[:8]
                cumulative = miners[uid].cumulative_weight
                logger.info(
                    f"  UID {uid} ({hotkey}...): "
                    f"Final={weight:.6f}, "
                    f"Cumulative={cumulative:.6f}"
                )
        
        logger.info("=" * 80)
    
    def print_detailed_table(self, miners: Dict[int, MinerData], environments: list):
        """Print detailed scoring table with all metrics.
        
        Args:
            miners: Dict of all miners
            environments: List of environment names
        """
        logger.info("=" * 120)
        logger.info("DETAILED SCORING TABLE")
        logger.info("=" * 120)
        
        # Build header
        header_parts = ["UID", "Hotkey"]
        for env in sorted(environments):
            header_parts.append(f"{env[:6]}")
        
        # Add layer columns
        max_layer = max(
            (max(m.layer_weights.keys()) if m.layer_weights else 0)
            for m in miners.values()
        )
        for layer in range(1, max_layer + 1):
            header_parts.append(f"L{layer}")
        
        header_parts.extend(["Total", "Valid"])
        
        logger.info(" | ".join(header_parts))
        logger.info("-" * 120)
        
        # Sort miners by final weight
        sorted_miners = sorted(
            miners.values(),
            key=lambda m: m.normalized_weight,
            reverse=True
        )
        
        # Print each miner row
        for miner in sorted_miners:
            row_parts = [
                f"{miner.uid:3d}",
                f"{miner.hotkey[:8]:8s}"
            ]
            
            # Environment scores
            for env in sorted(environments):
                if env in miner.env_scores:
                    score = miner.env_scores[env]
                    if score.is_valid:
                        row_parts.append(f"{score.avg_score:.3f}")
                    else:
                        row_parts.append("  -  ")
                else:
                    row_parts.append("  -  ")
            
            # Layer weights
            for layer in range(1, max_layer + 1):
                weight = miner.layer_weights.get(layer, 0.0)
                row_parts.append(f"{weight:.4f}")
            
            # Total weight and validity
            row_parts.append(f"{miner.normalized_weight:.6f}")
            row_parts.append("✓" if miner.is_valid_for_scoring() else "✗")
            
            logger.info(" | ".join(row_parts))
        
        logger.info("=" * 120)