"""
Scorer Configuration

Central configuration for the scoring algorithm.
All parameters are defined as constants for clarity and maintainability.
"""

from typing import Dict, Any


class ScorerConfig:
    """Configuration for the four-stage scoring algorithm."""
    
    # Stage 2: Pareto Frontier Anti-Plagiarism
    ERROR_RATE_REDUCTION: float = 0.2  # 20% error rate reduction threshold
    """
    Error rate reduction threshold for Pareto dominance.
    
    Formula: required_score = 0.2 + 0.8 * prior_score
    
    Examples:
    - prior_score=0.5 → error_rate=0.5 → required=0.6 (need 20% error reduction)
    - prior_score=0.9 → error_rate=0.1 → required=0.92 (need 20% error reduction)
    """
    
    MIN_IMPROVEMENT: float = 0.02
    """
    Minimum improvement required for later miner to beat earlier miner.
    
    Later miner must achieve: score_later > score_earlier + MIN_IMPROVEMENT
    This prevents random fluctuations in high-score regions from allowing
    plagiarism to beat originals.
    
    Example: If prior=0.9979 and MIN_IMPROVEMENT=0.02,
    later miner needs 0.9979 + 0.02 = 1.0179 (capped by MAX_IMPROVEMENT).
    
    Recommended value: 0.02
    """
    
    MAX_IMPROVEMENT: float = 0.1
    """
    Maximum improvement threshold cap.
    
    Caps the required score threshold to prevent unreasonably high values.
    The final threshold is clamped to: prior_score + MAX_IMPROVEMENT
    
    This ensures that even with low prior scores, the threshold remains reasonable.
    
    Recommended value: 0.1 (allows maximum 10% improvement)
    """
    
    SCORE_PRECISION: int = 3
    """Number of decimal places for score comparison (avoid floating point issues)."""
    
    # Stage 3: Subset Scoring
    MAX_LAYERS: int = 6
    """Maximum number of layers to evaluate. Due to exponential growth, 6 layers provide sufficient differentiation (2^5 = 32x difference between L1 and L6)."""
    
    SUBSET_WEIGHT_BASE: int = 1
    """Base weight multiplier for subset layers (N for L1, N*2 for L2, N*4 for L3, etc.)."""
    
    SUBSET_WEIGHT_EXPONENT: int = 2
    """Exponent base for layer weights (layer_weight = N * base^(layer-1))."""
    
    DECAY_FACTOR: float = 0.5
    """
    Rank-based decay factor for score_proportional weighting.
    
    Applied as: adjusted_score = score × decay_factor^(rank - 1)
    - Rank 1: score × 1.0
    - Rank 2: score × decay_factor^1
    - Rank 3: score × decay_factor^2
    
    Set to 1.0 to disable decay (all ranks weighted equally).
    Set to 0.5 for exponential decay (each rank gets 50% of previous).
    """
    
    # Stage 4: Weight Normalization
    MIN_WEIGHT_THRESHOLD: float = 0.01
    """Minimum weight threshold (1%). Miners below this are set to 0."""
    
    # Stage 1: Data Collection
    MIN_COMPLETENESS: float = 0.9
    """Minimum sample completeness required."""
    
    # Environment Score Normalization
    # Format: env_name -> (min_score, max_score)
    # Scores will be normalized to [0, 1] range: (score - min) / (max - min)
    ENV_SCORE_RANGES: Dict[str, tuple] = {
        'agentgym:sciworld': (-100, 100.0)  # sciworld 分数范围 0-100
    }
    
    # Database & Storage
    SCORE_RECORD_TTL_DAYS: int = 30
    """TTL for score_snapshots table (in days)."""
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Export configuration as dictionary for storage in snapshots."""
        return {
            'error_rate_reduction': cls.ERROR_RATE_REDUCTION,
            'min_improvement': cls.MIN_IMPROVEMENT,
            'max_improvement': cls.MAX_IMPROVEMENT,
            'score_precision': cls.SCORE_PRECISION,
            'max_layers': cls.MAX_LAYERS,
            'subset_weight_base': cls.SUBSET_WEIGHT_BASE,
            'subset_weight_exponent': cls.SUBSET_WEIGHT_EXPONENT,
            'decay_factor': cls.DECAY_FACTOR,
            'min_weight_threshold': cls.MIN_WEIGHT_THRESHOLD,
            'min_completeness': cls.MIN_COMPLETENESS,
        }
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters."""
        assert 0.0 <= cls.ERROR_RATE_REDUCTION <= 1.0, "ERROR_RATE_REDUCTION must be in [0, 1]"
        assert cls.MIN_IMPROVEMENT >= 0.0, "MIN_IMPROVEMENT must be non-negative"
        assert cls.MAX_IMPROVEMENT >= cls.MIN_IMPROVEMENT, "MAX_IMPROVEMENT must be >= MIN_IMPROVEMENT"
        assert cls.SCORE_PRECISION >= 0, "SCORE_PRECISION must be non-negative"
        assert cls.SUBSET_WEIGHT_BASE > 0, "SUBSET_WEIGHT_BASE must be positive"
        assert cls.SUBSET_WEIGHT_EXPONENT >= 2, "SUBSET_WEIGHT_EXPONENT must be >= 2"
        assert 0.0 <= cls.DECAY_FACTOR <= 1.0, "DECAY_FACTOR must be in [0, 1]"
        assert 0.0 <= cls.MIN_WEIGHT_THRESHOLD <= 1.0, "MIN_WEIGHT_THRESHOLD must be in [0, 1]"
        assert 0.0 <= cls.MIN_COMPLETENESS <= 1.0, "MIN_COMPLETENESS must be in [0, 1]"


# Validate configuration on import
ScorerConfig.validate()