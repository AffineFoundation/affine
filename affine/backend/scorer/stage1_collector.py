"""
Stage 1: Data Collection and Average Score Calculation

Collects sample data for all valid miners and calculates average scores
per environment with completeness validation.
"""

import logging
from typing import Dict, List, Any
from models import (
    MinerData,
    EnvScore,
    Stage1Output,
)
from config import ScorerConfig

from affine.core.setup import logger


class Stage1Collector:
    """Stage 1: Data Collection and Average Score Calculation.
    
    Responsibilities:
    1. Parse scoring data from API response
    2. Calculate average scores per environment for each miner
    3. Validate sample completeness (must be >= 95% of required range)
    4. Build MinerData objects with environment scores
    """
    
    def __init__(self, config: ScorerConfig = ScorerConfig):
        """Initialize Stage 1 collector.
        
        Args:
            config: Scorer configuration (defaults to global config)
        """
        self.config = config
        self.min_completeness = config.MIN_COMPLETENESS
    
    def collect(
        self,
        scoring_data: Dict[str, Any],
        environments: List[str]
    ) -> Stage1Output:
        """Collect and process scoring data for all miners.
        
        Args:
            scoring_data: Response from /api/v1/samples/scoring endpoint
                Format: {
                    "uid": {
                        "hotkey": "5...",
                        "model_revision": "...",
                        "env": {
                            "affine:sat": {
                                "samples": [...],
                                "total_count": 500,
                                "completed_count": 498,
                                "completeness": 0.996
                            }
                        }
                    }
                }
            environments: List of environment names participating in scoring
            
        Returns:
            Stage1Output containing processed miner data
        """
        logger.info(f"Stage 1: Starting data collection for {len(scoring_data)} miners")
        
        miners: Dict[int, MinerData] = {}
        valid_count = 0
        invalid_count = 0
        
        for uid_str, miner_entry in scoring_data.items():
            try:
                uid = int(uid_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid UID format: {uid_str}")
                continue
            
            # Extract miner metadata
            hotkey = miner_entry.get('hotkey', '')
            model_revision = miner_entry.get('model_revision', '')
            first_block = miner_entry.get('first_block', 0)
            env_data = miner_entry.get('env', {})
            
            if not hotkey or not model_revision:
                logger.warning(f"UID {uid}: Missing hotkey or model_revision")
                invalid_count += 1
                continue
            
            # Create MinerData object
            miner = MinerData(
                uid=uid,
                hotkey=hotkey,
                model_revision=model_revision,
                first_block=first_block
            )
            
            # Process each environment
            for env_name in environments:
                env_info = env_data.get(env_name, {})
                
                if not env_info:
                    # Environment data missing
                    miner.env_scores[env_name] = EnvScore(
                        avg_score=0.0,
                        sample_count=0,
                        completeness=0.0,
                        is_valid=False
                    )
                    continue
                
                # Extract environment data
                samples = env_info.get('samples', [])
                total_count = env_info.get('total_count', 0)
                completed_count = env_info.get('completed_count', 0)
                completeness = env_info.get('completeness', 0.0)
                
                # Calculate average score
                if samples:
                    scores = [s.get('score', 0.0) for s in samples]
                    avg_score = sum(scores) / len(scores)
                else:
                    avg_score = 0.0
                
                # Validate completeness
                is_valid = completeness >= self.min_completeness
                
                # Store environment score
                miner.env_scores[env_name] = EnvScore(
                    avg_score=avg_score,
                    sample_count=completed_count,
                    completeness=completeness,
                    is_valid=is_valid
                )
                
                if not is_valid:
                    logger.debug(
                        f"UID {uid} env {env_name}: Insufficient completeness "
                        f"({completeness:.2%} < {self.min_completeness:.0%})"
                    )
            
            # Check if miner has at least one valid environment
            if miner.is_valid_for_scoring():
                valid_count += 1
            else:
                invalid_count += 1
                logger.info(
                    f"UID {uid} ({hotkey[:8]}...): No valid environments "
                    f"(completeness < {self.min_completeness:.0%})"
                )
            
            miners[uid] = miner
        
        logger.info(
            f"Stage 1: Completed data collection - "
            f"Valid: {valid_count}, Invalid: {invalid_count}"
        )
        
        return Stage1Output(
            miners=miners,
            environments=environments,
            valid_count=valid_count,
            invalid_count=invalid_count
        )
    
    def print_summary(self, output: Stage1Output):
        """Print Stage 1 summary for debugging.
        
        Args:
            output: Stage 1 output data
        """
        logger.info("=" * 80)
        logger.info("STAGE 1 SUMMARY: Data Collection")
        logger.info("=" * 80)
        logger.info(f"Total Miners: {len(output.miners)}")
        logger.info(f"Valid Miners: {output.valid_count}")
        logger.info(f"Invalid Miners: {output.invalid_count}")
        logger.info(f"Environments: {len(output.environments)}")
        logger.info("")
        
        # Print per-environment statistics
        env_stats = {}
        for env in output.environments:
            env_stats[env] = {
                'valid': 0,
                'invalid': 0,
                'total_samples': 0,
                'avg_completeness': []
            }
        
        for miner in output.miners.values():
            for env, score in miner.env_scores.items():
                if score.is_valid:
                    env_stats[env]['valid'] += 1
                else:
                    env_stats[env]['invalid'] += 1
                env_stats[env]['total_samples'] += score.sample_count
                env_stats[env]['avg_completeness'].append(score.completeness)
        
        logger.info("Per-Environment Statistics:")
        for env, stats in sorted(env_stats.items()):
            avg_comp = sum(stats['avg_completeness']) / len(stats['avg_completeness']) if stats['avg_completeness'] else 0
            logger.info(
                f"  {env}: "
                f"Valid={stats['valid']}, "
                f"Invalid={stats['invalid']}, "
                f"Samples={stats['total_samples']}, "
                f"AvgCompleteness={avg_comp:.2%}"
            )
        
        logger.info("=" * 80)