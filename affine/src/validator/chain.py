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


def normalize_weights(weights: List[float], max_weight_limit: float = 0.1) -> List[float]:
    """
    Normalize weights to sum to 1 and max value not exceeding limit
    
    Args:
        weights: Raw weight list
        max_weight_limit: Maximum weight limit
        
    Returns:
        Normalized weight list
    """
    import numpy as np
    
    weights_array = np.array(weights, dtype=np.float32)
    
    # If all zeros, return uniform distribution
    if weights_array.sum() == 0:
        return [1.0 / len(weights)] * len(weights)
    
    # First normalize to sum to 1
    normalized = weights_array / weights_array.sum()
    
    # If max value doesn't exceed limit, return directly
    if normalized.max() <= max_weight_limit:
        return normalized.tolist()
    
    # Need to clip weights exceeding limit
    # Sort to find weights that need clipping
    sorted_indices = np.argsort(weights_array)[::-1]
    sorted_weights = weights_array[sorted_indices]
    
    # Calculate clipping threshold
    cumsum = 0.0
    cutoff_idx = 0
    for i, w in enumerate(sorted_weights):
        if w / weights_array.sum() <= max_weight_limit:
            cutoff_idx = i
            break
        cumsum += w
    
    # Calculate clipping value
    if cutoff_idx > 0:
        remaining_mass = 1.0 - (cutoff_idx * max_weight_limit)
        remaining_weights_sum = weights_array.sum() - cumsum
        
        if remaining_weights_sum > 0:
            scale = remaining_mass / remaining_weights_sum
        else:
            scale = 0.0
        
        # Apply clipping
        result = np.zeros_like(weights_array)
        for i in range(len(weights_array)):
            if weights_array[i] / weights_array.sum() > max_weight_limit:
                result[i] = max_weight_limit
            else:
                result[i] = weights_array[i] * scale
        
        # Re-normalize
        result = result / result.sum()
        return result.tolist()
    
    return normalized.tolist()


def apply_burn(
    uids: List[int],
    weights: List[float],
    burn_percentage: float
) -> Tuple[List[int], List[float]]:
    """
    Apply burn mechanism - force allocate percentage of weight to UID 0
    
    Args:
        uids: UID list
        weights: Weight list
        burn_percentage: Burn percentage (0.0 - 1.0)
        
    Returns:
        (uids, weights) after applying burn
    """
    import numpy as np
    
    if burn_percentage <= 0:
        return uids, weights
    
    burn_percentage = min(burn_percentage, 1.0)
    
    # Calculate amount to burn
    weights_array = np.array(weights, dtype=np.float32)
    total_weight = weights_array.sum()
    burn_amount = total_weight * burn_percentage
    
    # Reduce all weights proportionally
    remaining_weights = weights_array * (1.0 - burn_percentage)
    
    # Ensure UID 0 is in the list
    if 0 in uids:
        idx = uids.index(0)
        remaining_weights[idx] += burn_amount
    else:
        # UID 0 not in list, add it
        uids = [0] + list(uids)
        remaining_weights = np.concatenate([[burn_amount], remaining_weights])
    
    return uids, remaining_weights.tolist()


def convert_weights_to_u16(
    uids: List[int],
    weights: List[float]
) -> Tuple[List[int], List[int]]:
    """
    Convert weights to uint16 format for chain submission
    
    Args:
        uids: UID list
        weights: Normalized weight list (sum to 1)
        
    Returns:
        (valid uids, uint16 format weights)
    """
    if len(uids) != len(weights):
        raise ValueError(f"UID and weight count mismatch: {len(uids)} vs {len(weights)}")
    
    # Normalize to ensure sum is 1
    import numpy as np
    weights_array = np.array(weights, dtype=np.float32)
    
    if weights_array.sum() == 0:
        logger.warning("All weights are zero, cannot convert")
        return [], []
    
    normalized = weights_array / weights_array.sum()
    
    # Convert to uint16
    uint16_weights = []
    valid_uids = []
    
    for uid, weight in zip(uids, normalized):
        uint16_val = round(float(weight) * U16_MAX)
        
        # Filter out 0 values
        if uint16_val > 0:
            valid_uids.append(int(uid))
            uint16_weights.append(uint16_val)
    
    logger.debug(f"Converted to uint16: {len(valid_uids)} valid weights")
    return valid_uids, uint16_weights


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