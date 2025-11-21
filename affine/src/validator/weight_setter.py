"""
Weight Setter - Blockchain Weight Management

Handles setting weights on the Bittensor blockchain with retry logic and error handling.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from affine.core.setup import logger


@dataclass
class WeightSetResult:
    """Result of weight setting operation."""
    success: bool
    uids: List[int]
    weights: List[float]
    block_number: Optional[int] = None
    error_message: Optional[str] = None
    attempts: int = 0
    timestamp: float = 0.0


class WeightSetter:
    """
    Handles setting weights on the Bittensor blockchain.
    
    Provides:
    - Retry logic for failed attempts
    - Rate limiting to avoid spamming chain
    - Error handling and logging
    """
    
    def __init__(
        self,
        wallet=None,
        netuid: int = 1,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        min_interval: float = 300.0,  # 5 minutes between weight sets
    ):
        """
        Initialize WeightSetter.
        
        Args:
            wallet: Bittensor wallet instance
            netuid: Subnet UID
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            min_interval: Minimum interval between weight sets
        """
        self.wallet = wallet
        self.netuid = netuid
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.min_interval = min_interval
        
        self.last_set_at: Optional[float] = None
        self.total_sets = 0
        self.failed_sets = 0
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """
        Normalize weights to sum to 1.0.
        
        Args:
            weights: Raw weight values
            
        Returns:
            Normalized weights summing to 1.0
        """
        total = sum(weights)
        if total == 0:
            return [0.0] * len(weights)
        return [w / total for w in weights]
    
    def _validate_inputs(self, uids: List[int], weights: List[float]) -> Tuple[bool, str]:
        """
        Validate UIDs and weights.
        
        Args:
            uids: List of UIDs
            weights: List of weights
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not uids:
            return False, "Empty UID list"
        
        if len(uids) != len(weights):
            return False, f"UID count ({len(uids)}) != weight count ({len(weights)})"
        
        if any(uid < 0 for uid in uids):
            return False, "Negative UID found"
        
        if any(w < 0 for w in weights):
            return False, "Negative weight found"
        
        if len(set(uids)) != len(uids):
            return False, "Duplicate UIDs found"
        
        return True, ""
    
    def can_set_weights(self) -> Tuple[bool, float]:
        """
        Check if enough time has passed since last weight set.
        
        Returns:
            Tuple of (can_set, seconds_until_allowed)
        """
        if self.last_set_at is None:
            return True, 0.0
        
        elapsed = time.time() - self.last_set_at
        remaining = self.min_interval - elapsed
        
        if remaining <= 0:
            return True, 0.0
        
        return False, remaining
    
    async def set_weights(
        self,
        uids: List[int],
        weights: List[float],
        normalize: bool = True,
        force: bool = False,
    ) -> WeightSetResult:
        """
        Set weights on blockchain.
        
        Args:
            uids: List of miner UIDs
            weights: List of corresponding weights
            normalize: Whether to normalize weights to sum to 1.0
            force: Force weight set even if within min_interval
            
        Returns:
            WeightSetResult with operation details
        """
        # Check rate limit
        if not force:
            can_set, remaining = self.can_set_weights()
            if not can_set:
                logger.warning(f"Weight set rate limited. Wait {remaining:.1f}s")
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message=f"Rate limited. Wait {remaining:.1f}s",
                )
        
        # Validate inputs
        is_valid, error_msg = self._validate_inputs(uids, weights)
        if not is_valid:
            logger.error(f"Invalid weight inputs: {error_msg}")
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message=error_msg,
            )
        
        # Normalize if requested
        if normalize:
            weights = self._normalize_weights(weights)
        
        logger.info(
            f"Setting weights for {len(uids)} miners "
            f"(total: {sum(weights):.4f})"
        )
        
        # Attempt to set weights with retries
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = await self._do_set_weights(uids, weights)
                
                if result.success:
                    self.last_set_at = time.time()
                    self.total_sets += 1
                    result.attempts = attempt
                    result.timestamp = self.last_set_at
                    
                    logger.info(
                        f"Weights set successfully on attempt {attempt}. "
                        f"Block: {result.block_number}"
                    )
                    return result
                else:
                    last_error = result.error_message
                    logger.warning(
                        f"Weight set attempt {attempt} failed: {last_error}"
                    )
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"Weight set attempt {attempt} error: {e}")
            
            if attempt < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)
        
        # All retries failed
        self.failed_sets += 1
        return WeightSetResult(
            success=False,
            uids=uids,
            weights=weights,
            error_message=f"Failed after {self.max_retries} attempts: {last_error}",
            attempts=self.max_retries,
        )
    
    async def _do_set_weights(
        self,
        uids: List[int],
        weights: List[float],
    ) -> WeightSetResult:
        """
        Actually set weights on chain.
        
        Args:
            uids: List of UIDs
            weights: List of weights
            
        Returns:
            WeightSetResult
        """
        if self.wallet is None:
            logger.error("No wallet configured")
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message="No wallet configured",
            )
        
        try:
            # Import bittensor
            import bittensor as bt
            
            # Get subtensor connection
            subtensor = bt.subtensor()
            
            # Convert to tensors
            import torch
            uid_tensor = torch.tensor(uids, dtype=torch.int64)
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            
            # Set weights on chain
            success, msg = subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=uid_tensor,
                weights=weight_tensor,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            
            if success:
                # Get current block
                block_number = subtensor.get_current_block()
                
                return WeightSetResult(
                    success=True,
                    uids=uids,
                    weights=weights,
                    block_number=block_number,
                )
            else:
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message=str(msg),
                )
        
        except ImportError as e:
            logger.error(f"bittensor not installed: {e}")
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message="bittensor not installed",
            )
        
        except Exception as e:
            logger.error(f"Error setting weights on chain: {e}")
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message=str(e),
            )
    
    def get_metrics(self) -> Dict:
        """
        Get weight setter metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "total_sets": self.total_sets,
            "failed_sets": self.failed_sets,
            "success_rate": (
                self.total_sets / (self.total_sets + self.failed_sets)
                if (self.total_sets + self.failed_sets) > 0
                else 0.0
            ),
            "last_set_at": self.last_set_at,
            "netuid": self.netuid,
        }


async def test_weight_setter():
    """Test weight setter (without actually setting weights)."""
    # Create weight setter without wallet
    ws = WeightSetter(wallet=None, netuid=1)
    
    # Test validation
    valid, msg = ws._validate_inputs([1, 2, 3], [0.3, 0.3, 0.4])
    print(f"Validation: {valid}, {msg}")
    
    # Test normalization
    normalized = ws._normalize_weights([1.0, 2.0, 3.0])
    print(f"Normalized: {normalized}")
    
    # Test rate limiting
    can_set, remaining = ws.can_set_weights()
    print(f"Can set: {can_set}, remaining: {remaining}")
    
    # Test metrics
    metrics = ws.get_metrics()
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(test_weight_setter())