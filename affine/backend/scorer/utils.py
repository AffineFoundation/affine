"""
Scorer Utility Functions

Helper functions for the scoring algorithm.
"""

from typing import List, Dict, Set, Tuple
from itertools import combinations
import math


def generate_all_subsets(envs: List[str]) -> Dict[str, Dict[str, any]]:
    """Generate all possible subsets (environment combinations) with layer information.
    
    Args:
        envs: List of environment names
        
    Returns:
        Dict mapping subset_key to subset metadata:
        {
            "L1_sat": {
                "layer": 1,
                "envs": ["sat"],
                "key": "L1_sat"
            },
            "L2_sat_abd": {
                "layer": 2,
                "envs": ["sat", "abd"],
                "key": "L2_sat_abd"
            },
            ...
        }
    """
    subsets = {}
    n = len(envs)
    
    # Generate all combinations for each layer
    for layer in range(1, n + 1):
        for env_combo in combinations(envs, layer):
            # Sort environments alphabetically for consistent keys
            sorted_envs = sorted(env_combo)
            
            # Create subset key: L{layer}_{env1}_{env2}_{...}
            subset_key = f"L{layer}_{'_'.join(sorted_envs)}"
            
            subsets[subset_key] = {
                "layer": layer,
                "envs": sorted_envs,
                "key": subset_key
            }
    
    return subsets


def calculate_layer_weights(n_envs: int, base: int = 2) -> Dict[int, float]:
    """Calculate weight for each layer based on exponential growth.
    
    Layer weight = N × base^(layer-1)
    
    Args:
        n_envs: Number of environments
        base: Exponent base (default: 2)
        
    Returns:
        Dict mapping layer number to total layer weight:
        {1: N, 2: N*2, 3: N*4, 4: N*8, ...}
    """
    layer_weights = {}
    for layer in range(1, n_envs + 1):
        layer_weights[layer] = n_envs * (base ** (layer - 1))
    return layer_weights


def calculate_subset_weights(
    subsets: Dict[str, Dict[str, any]],
    layer_weights: Dict[int, float]
) -> Dict[str, float]:
    """Calculate individual subset weights by distributing layer weights equally.
    
    Each subset in a layer gets: layer_weight / num_subsets_in_layer
    
    Args:
        subsets: Dict of subset metadata
        layer_weights: Dict mapping layer to total weight
        
    Returns:
        Dict mapping subset_key to individual weight
    """
    # Count subsets per layer
    layer_counts = {}
    for subset_info in subsets.values():
        layer = subset_info["layer"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    # Distribute layer weights equally among subsets
    subset_weights = {}
    for subset_key, subset_info in subsets.items():
        layer = subset_info["layer"]
        layer_weight = layer_weights[layer]
        num_subsets = layer_counts[layer]
        
        subset_weights[subset_key] = layer_weight / num_subsets
    
    return subset_weights


def geometric_mean(values: List[float]) -> float:
    """Calculate geometric mean of a list of values.
    
    Formula: (∏ values)^(1/N)
    
    Args:
        values: List of numeric values
        
    Returns:
        Geometric mean, or 0.0 if any value is 0
    """
    if not values:
        return 0.0
    
    # If any value is 0, return 0 (penalizes poor performance)
    if any(v <= 0 for v in values):
        return 0.0
    
    # Calculate product and take Nth root
    n = len(values)
    product = 1.0
    for v in values:
        product *= v
    
    return product ** (1.0 / n)


def round_score(score: float, precision: int = 3) -> float:
    """Round score to specified decimal places.
    
    Args:
        score: Score value
        precision: Number of decimal places
        
    Returns:
        Rounded score
    """
    return round(score, precision)


def calculate_required_score(prior_score: float, error_rate_reduction: float = 0.2) -> float:
    """Calculate required score to beat prior based on error rate reduction.
    
    Formula: required_score = error_rate_reduction + (1 - error_rate_reduction) × prior_score
    
    This is derived from:
    - error_rate_old = 1 - prior_score
    - error_rate_required = error_rate_old × (1 - error_rate_reduction)
    - required_score = 1 - error_rate_required
    
    Simplified:
    - required_score = 1 - [(1 - prior_score) × (1 - error_rate_reduction)]
    - required_score = 1 - (1 - prior_score - error_rate_reduction + prior_score × error_rate_reduction)
    - required_score = error_rate_reduction + prior_score × (1 - error_rate_reduction)
    
    Args:
        prior_score: Score of the earlier miner
        error_rate_reduction: Required error rate reduction (default: 0.2 for 20%)
        
    Returns:
        Required score to dominate the prior miner
    """
    # Correct formula: error_rate_reduction + (1 - error_rate_reduction) × prior_score
    return error_rate_reduction + (1 - error_rate_reduction) * prior_score


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """Normalize weights to sum to 1.0.
    
    Args:
        weights: Dict mapping UID to raw weight
        
    Returns:
        Dict mapping UID to normalized weight (0.0 to 1.0)
    """
    total = sum(weights.values())
    
    if total == 0:
        return {uid: 0.0 for uid in weights}
    
    return {uid: w / total for uid, w in weights.items()}


def apply_min_threshold(
    weights: Dict[int, float],
    threshold: float = 0.01
) -> Dict[int, float]:
    """Set weights below threshold to 0.
    
    Args:
        weights: Dict mapping UID to weight
        threshold: Minimum weight threshold (default: 0.01 for 1%)
        
    Returns:
        Dict with sub-threshold weights set to 0
    """
    return {
        uid: (w if w >= threshold else 0.0)
        for uid, w in weights.items()
    }


def apply_burn_mechanism(
    weights: Dict[int, float],
    burn_percentage: float = 0.0,
    burn_uid: int = 0
) -> Tuple[Dict[int, float], float]:
    """Apply weight burning mechanism (allocate percentage to UID 0).
    
    Args:
        weights: Dict mapping UID to normalized weight
        burn_percentage: Percentage to burn (0.0 to 1.0)
        burn_uid: UID to receive burned weight (default: 0)
        
    Returns:
        Tuple of (updated_weights, burn_amount)
    """
    if burn_percentage <= 0:
        return weights.copy(), 0.0
    
    # Calculate total weight to burn
    total_weight = sum(weights.values())
    burn_amount = total_weight * burn_percentage
    
    # Create updated weights
    updated = weights.copy()
    updated[burn_uid] = updated.get(burn_uid, 0.0) + burn_amount
    
    return updated, burn_amount


def aggregate_by_layer(
    subset_weights: Dict[str, float]
) -> Dict[int, float]:
    """Aggregate subset weights by layer.
    
    Args:
        subset_weights: Dict mapping subset_key to weight contribution
        
    Returns:
        Dict mapping layer number to total weight
    """
    layer_totals = {}
    
    for subset_key, weight in subset_weights.items():
        # Extract layer from key (format: L{layer}_...)
        layer_str = subset_key.split('_')[0]  # "L3"
        layer = int(layer_str[1:])  # 3
        
        layer_totals[layer] = layer_totals.get(layer, 0.0) + weight
    
    return layer_totals


def format_score_table_row(
    uid: int,
    hotkey: str,
    env_scores: Dict[str, float],
    env_thresholds: Dict[str, float],
    env_samples: Dict[str, int],
    layer_weights: Dict[int, float],
    total_weight: float,
    is_valid: bool
) -> str:
    """Format a row for the score summary table.
    
    Args:
        uid: Miner UID
        hotkey: Miner hotkey (will be truncated)
        env_scores: Environment scores
        env_thresholds: Threshold upper bounds per environment
        env_samples: Sample counts per environment
        layer_weights: Weights by layer
        total_weight: Total cumulative weight
        is_valid: Whether miner is valid for scoring
        
    Returns:
        Formatted table row string
    """
    # Truncate hotkey
    hotkey_short = f"{hotkey[:8]}..."
    
    # Format environment columns
    env_cols = []
    for env in sorted(env_scores.keys()):
        score = env_scores.get(env, 0.0)
        threshold = env_thresholds.get(env, 0.0)
        samples = env_samples.get(env, 0)
        env_cols.append(f"{score:.3f}/{threshold:.3f}/{samples}")
    
    # Format layer weights
    layer_cols = [f"{layer_weights.get(i, 0.0):.4f}" for i in sorted(layer_weights.keys())]
    
    # Valid indicator
    valid_str = "✓" if is_valid else "✗"
    
    # Build row
    parts = [
        f"{uid:3d}",
        f"{hotkey_short:12s}",
        *env_cols,
        *layer_cols,
        f"{total_weight:.6f}",
        valid_str
    ]
    
    return " | ".join(parts)