"""
Simplified weight setter - removing fiber dependency

Handles weight normalization, burn mechanism, and chain setting.
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from affine.core.setup import logger
from affine.src.validator.chain import (
    load_keypair,
    get_substrate,
    query_chain,
    normalize_weights,
    apply_burn,
    convert_weights_to_u16,
    set_weights,
    WeightSetResult as ChainWeightSetResult,
    U16_MAX,
)


@dataclass
class WeightSetResult:
    """Weight setting result"""
    success: bool
    uids: List[int]
    weights: List[float]
    block_number: Optional[int] = None
    error_message: Optional[str] = None
    attempts: int = 0
    timestamp: float = 0.0


class WeightSetter:
    """
    Simplified weight setter
    
    Main functionality:
    1. Get weights and burn percentage from API
    2. Normalize weights (respecting chain's max_weight_limit)
    3. Apply burn mechanism (allocate weight to UID 0)
    4. Convert to uint16 format
    5. Submit to chain
    """
    
    def __init__(
        self,
        wallet_name: str,
        hotkey_name: str,
        netuid: int,
        network: str = "finney",
        network_address: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize weight setter
        
        Args:
            wallet_name: Wallet name
            hotkey_name: Hotkey name
            netuid: Subnet UID
            network: Network name (finney, test, local)
            network_address: Custom network address
            max_retries: Maximum retry attempts
            retry_delay: Retry delay in seconds
        """
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = netuid
        self.network = network
        self.network_address = network_address
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Statistics
        self.total_sets = 0
        self.failed_sets = 0
        
        # Load keypair
        try:
            self.keypair = load_keypair(wallet_name, hotkey_name)
            logger.debug(f"Loaded keypair: {wallet_name}/{hotkey_name}")
        except Exception as e:
            logger.error(f"Failed to load keypair: {e}")
            raise
    
    async def process_weights(
        self,
        uids: List[int],
        weights: List[float],
        burn_percentage: float = 0.0,
    ) -> Tuple[List[int], List[int]]:
        """
        Process weights: normalize + burn + convert to uint16
        
        Args:
            uids: UID list
            weights: Raw weight list
            burn_percentage: Burn percentage (0.0-1.0)
            
        Returns:
            (processed uids, uint16 format weights)
        """
        logger.debug(f"Processing {len(uids)} weights, burn: {burn_percentage:.1%}")
        
        # 1. Query chain parameters
        substrate = get_substrate(self.network, self.network_address)
        
        try:
            # Query max_weight_limit
            max_weight_limit_raw = query_chain(
                substrate,
                "SubtensorModule",
                "MaxWeightsLimit",
                [self.netuid],
                return_value=True
            )
            max_weight_limit = max_weight_limit_raw / U16_MAX if max_weight_limit_raw else 0.1
            
            # Query min_allowed_weights
            min_allowed_weights = query_chain(
                substrate,
                "SubtensorModule",
                "MinAllowedWeights",
                [self.netuid],
                return_value=True
            ) or 8
            
            logger.debug(
                f"Chain params: limit={max_weight_limit:.4f}, min={min_allowed_weights}"
            )
        
        finally:
            substrate.close()
        
        # 2. Normalize weights
        normalized_weights = normalize_weights(weights, max_weight_limit)
        logger.debug(f"Normalized weight sum: {sum(normalized_weights):.6f}")
        
        # 3. Apply burn mechanism
        if burn_percentage > 0:
            uids, normalized_weights = apply_burn(uids, normalized_weights, burn_percentage)
            logger.debug(f"Burn applied, UID 0: {normalized_weights[uids.index(0)]:.4f}")
        
        # 4. Filter non-zero weights
        import numpy as np
        weights_array = np.array(normalized_weights)
        non_zero_mask = weights_array > 0
        non_zero_uids = [uid for uid, mask in zip(uids, non_zero_mask) if mask]
        non_zero_weights = weights_array[non_zero_mask].tolist()
        
        logger.debug(f"Non-zero weights: {len(non_zero_uids)}/{len(uids)}")
        
        # 5. Check minimum weight count
        if len(non_zero_uids) < min_allowed_weights:
            raise ValueError(
                f"Non-zero weight count ({len(non_zero_uids)}) "
                f"less than minimum required ({min_allowed_weights})"
            )
        
        # 6. Convert to uint16 format
        final_uids, uint16_weights = convert_weights_to_u16(non_zero_uids, non_zero_weights)
        
        logger.debug(f"Processed {len(final_uids)} weights")
        
        return final_uids, uint16_weights
    
    async def set_weights(
        self,
        uids: List[int],
        weights: List[float],
        burn_percentage: float = 0.0,
    ) -> WeightSetResult:
        """
        Set weights on chain (with retry)
        
        Args:
            uids: UID list
            weights: Raw weight list
            burn_percentage: Burn percentage
            
        Returns:
            WeightSetResult
        """
        logger.info(f"Setting weights for {len(uids)} miners")
        
        # 1. Process weights once (outside retry loop)
        try:
            processed_uids, processed_weights = await self.process_weights(
                uids, weights, burn_percentage
            )
            
            if not processed_uids:
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message="No valid weights after processing",
                )
            
            # 2. Print weights before setting
            normalized_weights = [w / U16_MAX for w in processed_weights]
            
            logger.info("=" * 60)
            logger.info("Weights to be set on chain:")
            logger.info("=" * 60)
            for uid, weight in zip(processed_uids, normalized_weights):
                logger.info(f"  UID {uid:4d}: {weight:.6f}")
            logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"Failed to process weights: {e}")
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message=f"Weight processing failed: {e}",
            )
        
        # 3. Retry loop (only for chain submission)
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Get chain connection
                substrate = get_substrate(self.network, self.network_address)
                
                try:
                    # Query version key
                    version_key = query_chain(
                        substrate,
                        "SubtensorModule",
                        "WeightsVersionKey",
                        [self.netuid],
                        return_value=True
                    ) or 0
                    
                    logger.debug(f"Version key: {version_key}")
                    
                    # Submit to chain
                    result = await set_weights(
                        substrate=substrate,
                        keypair=self.keypair,
                        netuid=self.netuid,
                        uids=processed_uids,
                        weights=processed_weights,
                        version_key=version_key,
                        wait_for_inclusion=True,
                        wait_for_finalization=True,
                    )
                
                finally:
                    substrate.close()
                
                # Check result
                if result.success:
                    self.total_sets += 1
                    
                    logger.info(f"✅ Weights set successfully at block {result.block_number}")
                    
                    return WeightSetResult(
                        success=True,
                        uids=processed_uids,
                        weights=[w / U16_MAX for w in processed_weights],
                        block_number=result.block_number,
                        attempts=attempt,
                        timestamp=time.time(),
                    )
                else:
                    last_error = result.error_message
                    logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {last_error}")
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt}/{self.max_retries} exception: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # Wait before retry
            if attempt < self.max_retries:
                logger.info(f"Waiting {self.retry_delay}s before retry...")
                await asyncio.sleep(self.retry_delay)
        
        # All retries failed
        self.failed_sets += 1
        logger.error(f"❌ Setting failed after {self.max_retries} retries: {last_error}")
        
        return WeightSetResult(
            success=False,
            uids=uids,
            weights=weights,
            error_message=f"Failed after {self.max_retries} retries: {last_error}",
            attempts=self.max_retries,
        )
    
    def get_metrics(self) -> Dict:
        """Get statistics metrics"""
        return {
            "total_sets": self.total_sets,
            "failed_sets": self.failed_sets,
            "success_rate": (
                self.total_sets / (self.total_sets + self.failed_sets)
                if (self.total_sets + self.failed_sets) > 0
                else 0.0
            ),
            "netuid": self.netuid,
            "wallet": f"{self.wallet_name}/{self.hotkey_name}",
        }