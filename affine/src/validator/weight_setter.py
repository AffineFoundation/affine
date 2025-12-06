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
        1. Validate input
        2. Apply burn if needed (allocate to UID 0)
        3. Convert to uint16 using max-normalization
        
        Args:
            uids: List of miner UIDs
            weights: Weight values (can be unnormalized)
            burn_percentage: Burn percentage (0.0-1.0)
            
        Returns:
            Tuple of (processed_uids, uint16_weights) ready for chain submission
            
        Raises:
            ValueError: If validation fails
        """
        import numpy as np
        
        logger.debug(f"Processing {len(uids)} weights, burn={burn_percentage:.1%}")
        
        if len(uids) != len(weights):
            raise ValueError(f"UID count {len(uids)} != weight count {len(weights)}")
        
        if len(uids) == 0:
            raise ValueError("Empty UID list")
        
        weights_array = np.array(weights, dtype=np.float64)
        
        # Apply burn if needed
        if burn_percentage > 0 and burn_percentage <= 1.0:
            # Calculate burn amount
            burn_amount = weights_array.sum() * burn_percentage
            
            # Reduce all weights proportionally
            weights_array = weights_array * (1.0 - burn_percentage)
            
            # Add UID 0 with burn amount if not already present
            if 0 not in uids:
                uids = [0] + list(uids)
                weights_array = np.concatenate([[burn_amount], weights_array])
            else:
                # Add to existing UID 0
                idx = uids.index(0)
                weights_array[idx] += burn_amount
        
        # Convert to uint16 using max-normalization
        final_uids, uint16_weights = convert_weights_to_u16(
            list(uids),
            weights_array.tolist(),
        )
        
        if len(final_uids) == 0:
            raise ValueError("No valid weights after conversion")
        
        logger.info(
            f"Processed {len(final_uids)} weights for chain "
            f"(max={max(uint16_weights)}, sum={sum(uint16_weights)})"
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
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message="No valid weights after processing",
                    timestamp=time.time(),
                )
            
            # Print weights preview
            normalized_weights = [w / sum(processed_weights) for w in processed_weights]
            
            logger.info("=" * 60)
            logger.info("Weights to set on chain:")
            logger.info("=" * 60)
            for uid, n_weight, weight in zip(processed_uids, normalized_weights, processed_weights):
                logger.info(f"  UID {uid:4d}: {n_weight:.6f} {weight:.6f}")
            logger.info(f"Total: {sum(normalized_weights):.6f}")
            logger.info("=" * 60)
        
        except ValueError as e:
            # Parameter validation errors - don't retry
            logger.error(f"Invalid parameters: {e}")
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
                
                # Query version key from chain
                try:
                    result = substrate.query(
                        module="SubtensorModule",
                        storage_function="WeightsVersionKey",
                        params=[self.netuid],
                    )
                    version_key = result.value if result else 0
                except Exception as e:
                    logger.warning(f"Failed to query version key: {e}, using 0")
                    version_key = 0
                
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
        logger.error(f"❌ Failed to set weights after {self.max_retries} attempts: {last_error}")
        
        return WeightSetResult(
            success=False,
            uids=uids,
            weights=weights,
            error_message=f"Failed after {self.max_retries} retries: {last_error}",
            attempts=self.max_retries,
            timestamp=time.time(),
        )