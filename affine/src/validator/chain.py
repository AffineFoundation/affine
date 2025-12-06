#!/usr/bin/env python3
"""
Simplified chain interaction for Affine validator

Minimal implementation based on fiber/fiber/chain, keeping only essentials:
- Load keypairs from Bittensor wallet
- Connect to Substrate
- Convert weights using max-normalization (Bittensor standard)
- Submit weights to chain
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

from substrateinterface import SubstrateInterface
from bittensor import Keypair
from affine.core.setup import logger

# Bittensor constants
U16_MAX = 65535

# Network endpoints
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


def load_keypair(wallet_name: str, hotkey_name: str) -> Keypair:
    """
    Load hotkey keypair from Bittensor wallet directory.
    
    Matches fiber implementation: fiber/fiber/chain/chain_utils.py:load_hotkey_keypair
    
    Args:
        wallet_name: Wallet name
        hotkey_name: Hotkey name
        
    Returns:
        Keypair object
        
    Raises:
        ValueError: If keypair cannot be loaded
    """
    file_path = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys" / hotkey_name
    
    try:
        with open(file_path, "r") as f:
            keypair_data = json.load(f)
        
        keypair = Keypair.create_from_seed(keypair_data["secretSeed"])
        logger.info(f"Loaded keypair from {file_path}")
        return keypair
    
    except Exception as e:
        raise ValueError(f"Failed to load keypair {wallet_name}/{hotkey_name}: {e}")


def get_substrate(network: str = "finney", address: Optional[str] = None) -> SubstrateInterface:
    """
    Create Substrate connection.
    
    Matches fiber implementation: fiber/fiber/chain/interface.py:get_substrate
    
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
    
    logger.info(f"Connecting to chain: {endpoint}")
    
    return SubstrateInterface(
        ss58_format=42,
        use_remote_preset=True,
        url=endpoint,
    )


def convert_weights_to_u16(
    uids: List[int],
    weights: List[float],
) -> Tuple[List[int], List[int]]:
    """
    Convert weights to uint16 format using max-normalization (Bittensor standard).
    
    Matches fiber implementation: fiber/fiber/chain/weights.py:_normalize_and_quantize_weights
    
    Uses max-normalization: scales maximum weight to U16_MAX, maintaining relative ratios.
    This is the Bittensor standard and provides better precision than sum-normalization.
    
    Args:
        uids: List of miner UIDs
        weights: Raw weight values (unnormalized)
        
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
    
    weights_array = np.array(weights, dtype=np.float64)
    
    # Filter out non-positive weights first
    positive_mask = weights_array > 0
    if not positive_mask.any():
        logger.error("All weights are zero or negative")
        return [], []
    
    positive_uids = [uid for uid, mask in zip(uids, positive_mask) if mask]
    positive_weights = weights_array[positive_mask]
    
    # Max-normalization: scale max weight to U16_MAX
    max_weight = positive_weights.max()
    if max_weight == 0:
        logger.error("Max weight is zero")
        return [], []
    
    scaling_factor = U16_MAX / max_weight
    uint16_weights = np.round(positive_weights * scaling_factor).astype(np.uint16)
    
    # Filter out any zeros that appeared due to rounding
    nonzero_mask = uint16_weights > 0
    final_uids = [uid for uid, mask in zip(positive_uids, nonzero_mask) if mask]
    final_weights = uint16_weights[nonzero_mask].tolist()
    
    logger.debug(
        f"Converted to uint16 using max-normalization: "
        f"{len(final_uids)}/{len(uids)} weights, "
        f"max={max(final_weights)}, sum={sum(final_weights)}"
    )
    
    return final_uids, final_weights


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