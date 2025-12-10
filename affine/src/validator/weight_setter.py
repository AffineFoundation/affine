"""
Weight Setter

Handles weight processing and setting on chain.
"""

import bittensor as bt
from typing import List, Tuple, Dict
import numpy as np
import asyncio

from affine.core.setup import logger
from affine.utils.subtensor import get_subtensor

class WeightSetter:
    def __init__(self, wallet: bt.Wallet, netuid: int):
        self.wallet = wallet
        self.netuid = netuid

    async def process_weights(
        self,
        api_weights: Dict[str, Dict],
        burn_percentage: float = 0.0
    ) -> Tuple[List[int], List[float]]:
        """Process and normalize weights, applying burn if specified."""
        uids = []
        weights = []
        
        # Parse and filter valid weights
        for uid_str, weight_data in api_weights.items():
            try:
                uid = int(uid_str)
                weight = float(weight_data.get("weight", 0.0))
                if uid >= 0 and weight > 0:
                    uids.append(uid)
                    weights.append(weight)
            except (ValueError, TypeError):
                continue
                
        if not uids:
            return [], []

        # Normalize to sum = 1.0
        weights_array = np.array(weights, dtype=np.float64)
        weights_array = weights_array / weights_array.sum()
        
        # Apply burn: scale all by (1 - burn%), then UID 0 += burn%
        if burn_percentage > 0 and burn_percentage <= 1.0:
            weights_array *= (1.0 - burn_percentage)
            
            if 0 in uids:
                weights_array[uids.index(0)] += burn_percentage
            else:
                uids = [0] + uids
                weights_array = np.concatenate([[burn_percentage], weights_array])
                
        return uids, weights_array.tolist()

    async def set_weights(
        self,
        api_weights: Dict[str, Dict],
        burn_percentage: float = 0.0,
        max_retries: int = 3
    ) -> bool:
        """Set weights on chain with retry logic."""
        subtensor = await get_subtensor()
        uids, weights = await self.process_weights(api_weights, burn_percentage)
        
        if not uids:
            logger.warning("No valid weights to set")
            return False

        logger.info(f"Setting weights for {len(uids)} miners (burn={burn_percentage:.1%})")
        if burn_percentage > 0 and 0 in uids:
            logger.info(f"  UID 0 (burn): {weights[uids.index(0)]:.6f}")
            
        # Print uid:weight mapping
        logger.info("Weights to be set:")
        for uid, weight in zip(uids, weights):
            logger.info(f"  UID {uid:3d}: {weight:.6f}")

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                
                current_block = await subtensor.get_current_block()
                logger.info(f"Current block: {current_block}")
                
                success = await subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.netuid,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=True,
                    wait_for_finalization=True,
                )
                
                if success:
                    logger.info("✅ Weights set successfully (chain confirmed)")
                    return True
                else:
                    logger.error(f"❌ Chain rejected weight setting on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        logger.info("Retrying weight setting in 60 seconds...")
                        await asyncio.sleep(60)
                        continue
                    else:
                        logger.error("❌ All attempts failed")
                        return False

            except Exception as e:
                logger.error(f"Error setting weights on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying after error in 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                else:
                    return False
        
        return False