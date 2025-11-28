#!/usr/bin/env python3
"""
Simplified chain interaction utilities - replacing fiber library

Provides basic functionality for interacting with Bittensor chain:
- Load keypairs
- Connect to Substrate
- Query chain parameters
- Set weights
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from substrateinterface import SubstrateInterface
from bittensor import Keypair
from affine.core.setup import logger

# Chain constants
U16_MAX = 65535
U32_MAX = 4294967295


# Network endpoint mapping
NETWORK_ENDPOINTS = {
    "finney": "wss://entrypoint-finney.opentensor.ai:443",
    "test": "wss://test.finney.opentensor.ai:443/",
    "local": "ws://127.0.0.1:9944",
}


@dataclass
class WeightSetResult:
    """Weight setting result"""
    success: bool
    block_number: Optional[int] = None
    error_message: Optional[str] = None
    attempts: int = 0


def load_keypair(wallet_name: str, hotkey_name: str) -> Keypair:
    """
    Load hotkey keypair
    
    Args:
        wallet_name: Wallet name
        hotkey_name: Hotkey name
        
    Returns:
        Keypair object
    """
    file_path = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys" / hotkey_name
    
    try:
        with open(file_path, "r") as f:
            keypair_data = json.load(f)
        
        keypair = Keypair.create_from_seed(keypair_data["secretSeed"])
        logger.debug(f"Loaded keypair from {file_path}")
        return keypair
    
    except Exception as e:
        logger.error(f"Failed to load keypair: {e}")
        raise ValueError(f"Cannot load keypair {wallet_name}/{hotkey_name}: {e}")


def get_substrate(network: str = "finney", address: Optional[str] = None) -> SubstrateInterface:
    """
    Create Substrate connection
    
    Args:
        network: Network name (finney, test, local)
        address: Custom address (overrides network)
        
    Returns:
        SubstrateInterface object
    """
    if address:
        endpoint = address
    elif network in NETWORK_ENDPOINTS:
        endpoint = NETWORK_ENDPOINTS[network]
    else:
        raise ValueError(f"Unknown network: {network}")
    
    logger.debug(f"Connecting to chain: {endpoint}")
    
    substrate = SubstrateInterface(
        ss58_format=42,
        use_remote_preset=True,
        url=endpoint,
    )
    
    return substrate


def query_chain(
    substrate: SubstrateInterface,
    module: str,
    method: str,
    params: List[Any],
    return_value: bool = True
) -> Any:
    """
    Query chain data
    
    Args:
        substrate: Substrate interface
        module: Module name
        method: Method name
        params: Parameter list
        return_value: Whether to return .value
        
    Returns:
        Query result
    """
    try:
        result = substrate.query(module, method, params)
        return result.value if return_value else result
    except Exception as e:
        logger.error(f"Chain query failed {module}.{method}: {e}")
        raise


def normalize_weights(weights: List[float]) -> List[float]:
    """
    Normalize weights to sum to 1.0.
    
    Args:
        weights: Raw weight list (any positive values)
        
    Returns:
        Normalized weight list that sums to 1.0
    """
    import numpy as np
    
    weights_array = np.array(weights, dtype=np.float32)
    
    # Handle edge case: all zeros or empty
    if weights_array.sum() == 0 or len(weights) == 0:
        if len(weights) == 0:
            return []
        return [1.0 / len(weights)] * len(weights)
    
    # Normalize to sum = 1.0
    normalized = weights_array / weights_array.sum()
    
    return normalized.tolist()


def apply_burn(
    uids: List[int],
    weights: List[float],
    burn_percentage: float
) -> Tuple[List[int], List[float]]:
    """
    Apply burn mechanism - allocate a percentage of total weight to UID 0.
    
    This mechanism ensures UID 0 receives exactly burn_percentage of the total weight,
    while all other UIDs have their weights proportionally reduced.
    
    Logic:
        - UID 0 gets: burn_percentage (of total normalized weight)
        - Other UIDs get: their_weight * (1 - burn_percentage)
        - Result still sums to 1.0
    
    Args:
        uids: List of miner UIDs
        weights: List of weights (should sum to ~1.0 if normalized)
        burn_percentage: Percentage to allocate to UID 0 (0.0 - 1.0)
        
    Returns:
        Tuple of (updated_uids, updated_weights) with UID 0 included
    """
    import numpy as np
    
    if burn_percentage <= 0:
        return uids, weights
    
    burn_percentage = min(max(burn_percentage, 0.0), 1.0)
    
    # Convert to numpy for easier manipulation
    weights_array = np.array(weights, dtype=np.float32)
    
    # Normalize weights if not already normalized
    total = weights_array.sum()
    if abs(total - 1.0) > 0.01:
        weights_array = weights_array / total
    
    # Remove UID 0 if it already exists (we'll add it back with correct weight)
    if 0 in uids:
        uid_0_idx = uids.index(0)
        uids_list = uids[:uid_0_idx] + uids[uid_0_idx + 1:]
        weights_array = np.concatenate([
            weights_array[:uid_0_idx],
            weights_array[uid_0_idx + 1:]
        ])
    else:
        uids_list = list(uids)
    
    # Scale down all other weights
    scaled_weights = weights_array * (1.0 - burn_percentage)
    
    # Add UID 0 at the beginning with burn amount
    final_uids = [0] + uids_list
    final_weights = np.concatenate([[burn_percentage], scaled_weights])
    
    logger.debug(
        f"Applied burn: UID 0 gets {burn_percentage:.4f}, "
        f"others scaled by {1.0 - burn_percentage:.4f}"
    )
    
    return final_uids, final_weights.tolist()


def convert_weights_to_u16(
    uids: List[int],
    weights: List[float],
    validate_normalization: bool = True
) -> Tuple[List[int], List[int]]:
    """
    Convert normalized weights to uint16 format for chain submission.
    
    This is the final step before chain submission. Input weights should already
    be normalized (sum ≈ 1.0). This function only handles the conversion to uint16
    and filtering of zero weights.
    
    Args:
        uids: List of miner UIDs
        weights: Normalized weight list (should sum to ~1.0)
        validate_normalization: Whether to validate and fix normalization
        
    Returns:
        Tuple of (valid_uids, uint16_weights) with zero weights filtered out
        
    Raises:
        ValueError: If input validation fails
    """
    import numpy as np
    
    if len(uids) != len(weights):
        raise ValueError(
            f"UID count ({len(uids)}) doesn't match weight count ({len(weights)})"
        )
    
    if len(uids) == 0:
        logger.warning("Empty UID list, returning empty result")
        return [], []
    
    weights_array = np.array(weights, dtype=np.float32)
    
    # Check for zero sum
    if weights_array.sum() == 0:
        logger.error("All weights are zero, cannot convert to uint16")
        return [], []
    
    # Validate normalization if requested
    if validate_normalization:
        total = weights_array.sum()
        if abs(total - 1.0) > 0.01:
            weights_array = weights_array / total
    
    # Convert to uint16 by scaling to U16_MAX
    # Use round() to minimize rounding errors
    uint16_weights_float = weights_array * U16_MAX
    uint16_weights = np.round(uint16_weights_float).astype(np.uint16)
    
    # Filter out zero weights (can happen due to rounding)
    nonzero_mask = uint16_weights > 0
    valid_uids = [uid for uid, mask in zip(uids, nonzero_mask) if mask]
    valid_weights = uint16_weights[nonzero_mask].tolist()
    
    logger.debug(
        f"Converted to uint16: {len(valid_uids)}/{len(uids)} non-zero weights, "
        f"sum={sum(valid_weights)}/{U16_MAX}"
    )
    
    return valid_uids, valid_weights


async def set_weights(
    substrate: SubstrateInterface,
    keypair: Keypair,
    netuid: int,
    uids: List[int],
    weights: List[float],
    version_key: int = 0,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> WeightSetResult:
    """
    Set weights on chain
    
    Args:
        substrate: Substrate interface
        keypair: Validator keypair
        netuid: Subnet UID
        uids: UID list
        weights: Weight list (uint16 format)
        version_key: Version key
        wait_for_inclusion: Wait for inclusion
        wait_for_finalization: Wait for finalization
        
    Returns:
        WeightSetResult
    """
    try:
        logger.debug(f"Submitting to chain: {len(uids)} weights")
        
        # Create call
        call = substrate.compose_call(
            call_module="SubtensorModule",
            call_function="set_weights",
            call_params={
                "dests": uids,
                "weights": weights,
                "netuid": netuid,
                "version_key": version_key,
            },
        )
        
        # Create signed extrinsic
        extrinsic = substrate.create_signed_extrinsic(
            call=call,
            keypair=keypair,
        )
        
        # Submit to chain
        def submit():
            return substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
        
        # Execute blocking call in thread pool
        response = await asyncio.get_event_loop().run_in_executor(None, submit)
        
        if not wait_for_finalization and not wait_for_inclusion:
            return WeightSetResult(success=True)
        
        # Process events
        response.process_events()
        
        if response.is_success:
            block_number = response.block_number if hasattr(response, 'block_number') else None
            logger.debug(f"Chain submission successful at block: {block_number}")
            return WeightSetResult(
                success=True,
                block_number=block_number,
            )
        else:
            error_msg = str(response.error_message) if hasattr(response, 'error_message') else "Unknown error"
            logger.error(f"Chain submission failed: {error_msg}")
            return WeightSetResult(
                success=False,
                error_message=error_msg,
            )
    
    except Exception as e:
        logger.error(f"Exception setting weights: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return WeightSetResult(
            success=False,
            error_message=str(e),
        )