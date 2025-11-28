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
    apply_burn,
    convert_weights_to_u16,
    set_weights,
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
        self.last_set_at: Optional[float] = None
        
        # Cached substrate connection (reuse across calls)
        self._substrate: Optional[object] = None
        
        # Load keypair
        try:
            self.keypair = load_keypair(wallet_name, hotkey_name)
            logger.debug(f"Loaded keypair: {wallet_name}/{hotkey_name}")
        except Exception as e:
            logger.error(f"Failed to load keypair: {e}")
            raise
    
    def _get_substrate(self):
        """Get or create substrate connection (connection reuse)."""
        if self._substrate is None:
            self._substrate = get_substrate(self.network, self.network_address)
            logger.debug("Created new substrate connection")
        return self._substrate
    
    def _close_substrate(self):
        """Close substrate connection if open."""
        if self._substrate is not None:
            try:
                self._substrate.close()
                logger.debug("Closed substrate connection")
            except Exception as e:
                logger.warning(f"Error closing substrate: {e}")
            finally:
                self._substrate = None
    
    def __del__(self):
        """Cleanup substrate connection on deletion."""
        self._close_substrate()
    
    async def process_weights(
        self,
        uids: List[int],
        weights: List[float],
        burn_percentage: float = 0.0,
    ) -> Tuple[List[int], List[int]]:
        """
        Process weights for chain submission.
        
        Simplified workflow:
        1. Validate input weights (should already be normalized from API)
        2. Apply burn mechanism (allocate to UID 0)
        3. Convert to uint16 format
        
        Args:
            uids: List of miner UIDs
            weights: Weight list (expected to be normalized, sum ≈ 1.0)
            burn_percentage: Burn percentage (0.0-1.0)
            
        Returns:
            Tuple of (processed_uids, uint16_weights) ready for chain submission
            
        Raises:
            ValueError: If validation fails
        """
        import numpy as np
        
        logger.debug(f"Processing {len(uids)} weights, burn={burn_percentage:.1%}")
        
        # Validate input
        if len(uids) != len(weights):
            raise ValueError(f"UID count {len(uids)} != weight count {len(weights)}")
        
        if len(uids) == 0:
            raise ValueError("Empty UID list")
        
        weights_array = np.array(weights, dtype=np.float32)
        
        # Check if weights are already normalized (as expected from API)
        total_weight = weights_array.sum()
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Input weights sum to {total_weight:.6f}, not 1.0. "
                "API should return normalized weights. Normalizing now."
            )
            weights_array = weights_array / total_weight
        
        # Apply burn mechanism (modifies UID list and weights)
        if burn_percentage > 0:
            uids_list, weights_list = apply_burn(
                list(uids),
                weights_array.tolist(),
                burn_percentage
            )
        else:
            uids_list = list(uids)
            weights_list = weights_array.tolist()
        
        # Convert to uint16 format
        final_uids, uint16_weights = convert_weights_to_u16(
            uids_list,
            weights_list,
            validate_normalization=True
        )
        
        if len(final_uids) == 0:
            raise ValueError("No valid weights after uint16 conversion")
        
        logger.info(
            f"Processed {len(final_uids)} weights for chain "
            f"(sum={sum(uint16_weights)}/{U16_MAX})"
        )
        
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
        logger.info(f"Setting weights for {len(uids)} miners (burn={burn_percentage:.1%})")
        
        # Process weights once (outside retry loop to avoid redundant computation)
        try:
            processed_uids, processed_weights = await self.process_weights(
                uids, weights, burn_percentage
            )
            
            if not processed_uids:
                self.failed_sets += 1
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message="No valid weights after processing",
                    timestamp=time.time(),
                )
            
            # Print weights preview
            normalized_weights = [w / U16_MAX for w in processed_weights]
            
            logger.info("=" * 60)
            logger.info("Weights to set on chain:")
            logger.info("=" * 60)
            for uid, weight in zip(processed_uids, normalized_weights):
                logger.info(f"  UID {uid:4d}: {weight:.6f}")
            logger.info(f"Total: {sum(normalized_weights):.6f}")
            logger.info("=" * 60)
        
        except ValueError as e:
            # Parameter validation errors - don't retry
            logger.error(f"Invalid parameters: {e}")
            self.failed_sets += 1
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message=f"Invalid parameters: {e}",
                timestamp=time.time(),
            )
        
        except Exception as e:
            # Unexpected errors during processing
            logger.error(f"Failed to process weights: {e}", exc_info=True)
            self.failed_sets += 1
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message=f"Weight processing failed: {e}",
                timestamp=time.time(),
            )
        
        # Retry loop for chain submission only
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Use cached substrate connection
                substrate = self._get_substrate()
                
                # Query version key
                version_key = query_chain(
                    substrate,
                    "SubtensorModule",
                    "WeightsVersionKey",
                    [self.netuid],
                    return_value=True
                ) or 0
                
                logger.debug(f"Attempt {attempt}/{self.max_retries}, version_key={version_key}")
                
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
                
                # Check result
                if result.success:
                    self.total_sets += 1
                    self.last_set_at = time.time()
                    
                    logger.info(
                        f"✅ Weights set successfully at block {result.block_number} "
                        f"(attempt {attempt}/{self.max_retries})"
                    )
                    
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
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries} failed: {last_error}"
                    )
            
            except ConnectionError as e:
                # Network errors - retry
                last_error = str(e)
                logger.warning(f"Connection error on attempt {attempt}: {e}")
                # Reset connection on network error
                self._close_substrate()
            
            except ValueError as e:
                # Parameter errors - don't retry
                last_error = str(e)
                logger.error(f"Parameter error (won't retry): {e}")
                self.failed_sets += 1
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message=f"Parameter error: {last_error}",
                    attempts=attempt,
                    timestamp=time.time(),
                )
            
            except Exception as e:
                # Other errors - retry
                last_error = str(e)
                logger.error(f"Attempt {attempt}/{self.max_retries} error: {e}", exc_info=True)
            
            # Wait before retry (if not last attempt)
            if attempt < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)
        
        # All retries exhausted
        self.failed_sets += 1
        logger.error(f"❌ Failed to set weights after {self.max_retries} attempts: {last_error}")
        
        return WeightSetResult(
            success=False,
            uids=uids,
            weights=weights,
            error_message=f"Failed after {self.max_retries} retries: {last_error}",
            attempts=self.max_retries,
            timestamp=time.time(),
        )
    
    def get_metrics(self) -> Dict:
        """Get statistics metrics"""
        total_attempts = self.total_sets + self.failed_sets
        return {
            "total_sets": self.total_sets,
            "failed_sets": self.failed_sets,
            "total_attempts": total_attempts,
            "success_rate": (
                self.total_sets / total_attempts
                if total_attempts > 0
                else 0.0
            ),
            "last_set_at": self.last_set_at,
            "netuid": self.netuid,
            "wallet": f"{self.wallet_name}/{self.hotkey_name}",
        }