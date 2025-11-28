"""
Rank Display Module

Fetches and displays miner ranking information from the API,
using the same format as scorer's print_summary.
"""

import asyncio
from typing import Dict, Any, List, Optional
from affine.utils.api_client import create_api_client
from affine.core.setup import logger
from affine.core.miners import miners


async def fetch_latest_scores() -> Dict[str, Any]:
    """Fetch latest scores from API.
    
    Returns:
        Dict with block_number, calculated_at, and scores list
    """
    api_client = create_api_client()
    
    logger.debug("Fetching latest scores from API...")
    data = await api_client.get("/scores/latest?top=256")
    
    if isinstance(data, dict) and "success" in data and data.get("success") is False:
        error_msg = data.get("error", "Unknown API error")
        status_code = data.get("status_code", "unknown")
        logger.error(f"API returned error response: {error_msg} (status: {status_code})")
        raise RuntimeError(f"Failed to fetch scores: {error_msg}")
    
    return data


async def fetch_miner_scores_at_block(block_number: int) -> Dict[int, Dict[str, Any]]:
    """Fetch detailed miner scores for a specific block via API.
    
    Args:
        block_number: Block number to query
        
    Returns:
        Dict mapping UID to detailed score data
    """
    api_client = create_api_client()
    
    logger.debug(f"Fetching detailed scores for block {block_number}...")
    
    # Use the miner_scores endpoint to get detailed scoring data
    # This endpoint should return all the data needed for print_summary
    data = await api_client.get(f"/scores/latest?top=256")
    
    return data


async def fetch_environments() -> List[str]:
    """Fetch enabled environments from system config.
    
    Returns:
        List of environment names enabled for scoring
    """
    api_client = create_api_client()
    
    try:
        config = await api_client.get("/config/environments")
        
        if isinstance(config, dict):
            value = config.get("param_value")
            if isinstance(value, dict):
                # Filter environments where enabled_for_scoring=true
                enabled_envs = [
                    env_name for env_name, env_config in value.items()
                    if isinstance(env_config, dict) and env_config.get("enabled_for_scoring", False)
                ]
                
                if enabled_envs:
                    logger.debug(f"Fetched environments from API: {enabled_envs}")
                    return sorted(enabled_envs)
        
        logger.warning("Failed to parse environments config")
        return []
                
    except Exception as e:
        logger.error(f"Error fetching environments: {e}")
        return []


async def print_rank_table():
    """Fetch and print miner ranking table in scorer format.
    
    This function replicates the output format of scorer's print_detailed_table,
    but fetches data from the API instead of calculating from raw samples.
    """
    # Fetch scores and environments
    scores_data = await fetch_latest_scores()
    environments = await fetch_environments()
    
    if not scores_data or not scores_data.get('block_number'):
        print("No scores found")
        return
    
    block_number = scores_data.get("block_number")
    calculated_at = scores_data.get("calculated_at")
    scores_list = scores_data.get("scores", [])
    
    if not scores_list:
        print(f"No miners scored at block {block_number}")
        return
    
    # Print header
    print("=" * 180, flush=True)
    print(f"MINER RANKING TABLE - Block {block_number}", flush=True)
    print("=" * 180, flush=True)
    
    # Build header - Hotkey first, then UID, then Model, then First Block, then environments
    header_parts = ["Hotkey  ", "UID", "Model                    ", " FirstBlk "]
    
    # Format environment names - keep everything after ':'
    for env in environments:
        if ':' in env:
            env_display = env.split(':', 1)[1]
        else:
            env_display = env
        header_parts.append(f"{env_display:>11}")
    
    # Find all layers that have non-zero weights
    all_layers = set()
    for score in scores_list:
        scores_by_layer = score.get("scores_by_layer", {})
        for layer_key, weight in scores_by_layer.items():
            if weight > 0:
                # Extract layer number from "L3" format
                layer_num = int(layer_key[1:])
                all_layers.add(layer_num)
    
    active_layers = sorted(all_layers)
    
    for layer in active_layers:
        header_parts.append(f"{'L'+str(layer):>8}")
    
    header_parts.extend(["   Total ", "  Weight ", "V"])
    
    print(" | ".join(header_parts), flush=True)
    print("-" * 180, flush=True)
    
    # Print each miner row
    for score in scores_list:
        uid = score.get("uid")
        hotkey = score.get("miner_hotkey")
        model_revision = score.get("model_revision")
        model = score.get("model")
        first_block = score.get("first_block")
        overall_score = score.get("overall_score")
        scores_by_env = score.get("scores_by_env", {})
        scores_by_layer = score.get("scores_by_layer", {})
        total_samples = score.get("total_samples")
        is_eligible = score.get("is_eligible")
        
        model_display = model[:25]
        
        row_parts = [
            f"{hotkey[:8]:8s}",
            f"{uid:3d}",
            f"{model_display:25s}",
            f"{first_block:10d}"
        ]
        
        # Environment scores - show "score/count" format (score × 100, 2 decimals)
        for env in environments:
            if env in scores_by_env:
                env_data = scores_by_env[env]
                env_score = env_data.get("score", 0.0)
                sample_count = env_data.get("sample_count", 0)
                
                score_percent = env_score * 100
                score_str = f"{score_percent:.2f}/{sample_count}"
                row_parts.append(f"{score_str:>11}")
            else:
                row_parts.append(f"{'  -  ':>11}")
        
        # Layer weights - only for active layers
        for layer in active_layers:
            layer_key = f"L{layer}"
            weight = scores_by_layer.get(layer_key, 0.0)
            row_parts.append(f"{weight:>8.4f}")
        
        # Total (cumulative) and Weight (normalized)
        # Note: API doesn't provide cumulative_weight, so we use average_score as proxy
        average_score = score.get("average_score", 0.0)
        row_parts.append(f"{average_score:>9.4f}")  # Total: use average score
        row_parts.append(f"{overall_score:>9.6f}")  # Weight: normalized weight
        row_parts.append("✓" if is_eligible else "✗")
        
        print(" | ".join(row_parts), flush=True)
    
    print("=" * 180, flush=True)
    print(f"Total miners: {len(scores_list)}", flush=True)
    non_zero = len([s for s in scores_list if s.get("overall_score", 0.0) > 0])
    print(f"Active miners (weight > 0): {non_zero}", flush=True)
    print("=" * 180, flush=True)


async def get_rank_command():
    """Command handler for get-rank.
    
    Fetches score snapshot from API and displays ranking table.
    """
    try:
        await print_rank_table()
    except Exception as e:
        logger.error(f"Failed to fetch and display ranks: {e}", exc_info=True)
        print(f"Error: {e}")