"""
Weight Setter - Blockchain Weight Management

Handles setting weights on the Bittensor blockchain with proper normalization,
max weight limits, and retry logic using fiber.
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from fiber import SubstrateInterface
from fiber.chain import chain_utils, interface, weights
from fiber.chain.fetch_nodes import get_nodes_for_netuid

from affine.core.setup import logger


# Constants from Bittensor chain
U16_MAX = 65535
U32_MAX = 4294967295


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
    Handles setting weights on the Bittensor blockchain using fiber.
    
    Features:
    - Query chain parameters (min_allowed_weights, max_weight_limit)
    - Normalize weights with max weight limit enforcement
    - Convert to uint16 format for chain submission
    - Retry logic with timeout
    - Rate limiting
    """
    
    def __init__(
        self,
        wallet_name: str,
        hotkey_name: str,
        netuid: int,
        subtensor_network: str = "finney",
        subtensor_address: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        min_interval: float = 300.0,
        timeout: float = 120.0,
    ):
        """
        Initialize WeightSetter.
        
        Args:
            wallet_name: Wallet name for loading keypair
            hotkey_name: Hotkey name for loading keypair
            netuid: Subnet UID
            subtensor_network: Network name (finney, test, local)
            subtensor_address: Optional custom subtensor address
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            min_interval: Minimum interval between weight sets
            timeout: Timeout for weight setting operation
        """
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = netuid
        self.subtensor_network = subtensor_network
        self.subtensor_address = subtensor_address
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.min_interval = min_interval
        self.timeout = timeout
        
        self.last_set_at: Optional[float] = None
        self.total_sets = 0
        self.failed_sets = 0
        
        # Load keypair once during initialization
        try:
            self.keypair = chain_utils.load_hotkey_keypair(
                wallet_name=wallet_name,
                hotkey_name=hotkey_name
            )
            logger.info(f"Loaded keypair for {wallet_name}/{hotkey_name}")
        except Exception as e:
            logger.error(f"Failed to load keypair: {e}")
            raise
    
    def _get_substrate(self) -> SubstrateInterface:
        """Get substrate interface connection."""
        return interface.get_substrate(
            self.subtensor_network,
            self.subtensor_address
        )
    
    def _normalize_max_weight(
        self,
        x: np.ndarray,
        limit: float = 0.1
    ) -> np.ndarray:
        """
        Normalize array so sum(x) = 1 and max value <= limit.
        
        This is critical for chain acceptance - ensures no single weight
        dominates beyond the subnet's max_weight_limit.
        
        Args:
            x: Array to normalize
            limit: Maximum allowed value after normalization
            
        Returns:
            Normalized array
        """
        epsilon = 1e-7
        weights = x.copy()
        values = np.sort(weights)
        
        if x.sum() == 0 or len(x) * limit <= 1:
            return np.ones_like(x) / x.size
        
        estimation = values / values.sum()
        
        if estimation.max() <= limit:
            return weights / weights.sum()
        
        # Find cumulative sum
        cumsum = np.cumsum(estimation, 0)
        
        # Determine cutoff index
        estimation_sum = np.array(
            [(len(values) - i - 1) * estimation[i] for i in range(len(values))]
        )
        n_values = (estimation / (estimation_sum + cumsum + epsilon) < limit).sum()
        
        # Calculate cutoff scale
        cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
            1 - (limit * (len(estimation) - n_values))
        )
        cutoff = cutoff_scale * values.sum()
        
        # Apply cutoff
        weights[weights > cutoff] = cutoff
        
        return weights / weights.sum()
    
    def _convert_weights_and_uids_for_emit(
        self,
        uids: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Convert weights to uint16 representation for chain submission.
        
        Args:
            uids: Array of UIDs
            weights: Array of normalized weights (sum to 1.0)
            
        Returns:
            Tuple of (weight_uids, weight_vals) as lists of ints
        """
        uids = np.asarray(uids)
        weights = np.asarray(weights)
        
        # Validation
        if np.min(weights) < 0:
            raise ValueError(f"Negative weight found: {weights}")
        if np.min(uids) < 0:
            raise ValueError(f"Negative UID found: {uids}")
        if len(uids) != len(weights):
            raise ValueError(f"UID/weight length mismatch: {len(uids)} vs {len(weights)}")
        
        if np.sum(weights) == 0:
            logger.debug("All weights are zero, nothing to set")
            return [], []
        
        # Normalize to sum to 1
        weight_sum = float(np.sum(weights))
        normalized_weights = [float(w) / weight_sum for w in weights]
        
        logger.debug(f"Weight sum: {weight_sum}, normalized sum: {sum(normalized_weights)}")
        
        # Convert to uint16
        weight_vals = []
        weight_uids = []
        
        for uid, weight in zip(uids, normalized_weights):
            uint16_val = round(float(weight) * U16_MAX)
            
            # Filter zeros
            if uint16_val != 0:
                weight_vals.append(uint16_val)
                weight_uids.append(int(uid))
        
        logger.debug(f"Final UIDs: {weight_uids}, vals: {weight_vals}")
        
        return weight_uids, weight_vals
    
    def _process_weights_for_netuid(
        self,
        uids: np.ndarray,
        weights: np.ndarray,
        substrate: SubstrateInterface
    ) -> Tuple[List[int], List[float]]:
        """
        Process weights according to chain parameters.
        
        Queries chain for min_allowed_weights and max_weight_limit,
        then normalizes and converts weights appropriately.
        
        Args:
            uids: Array of UIDs
            weights: Array of raw weights
            substrate: Substrate interface
            
        Returns:
            Tuple of (node_ids, node_weights) ready for chain submission
        """
        # Ensure float32
        if not isinstance(weights, np.ndarray) or weights.dtype != np.float32:
            weights = weights.astype(np.float32)
        
        # Query chain parameters
        min_allowed_query = substrate.query("SubtensorModule", "MinAllowedWeights", [self.netuid])
        max_weight_query = substrate.query("SubtensorModule", "MaxWeightsLimit", [self.netuid])
        
        min_allowed_weights = min_allowed_query.value if min_allowed_query else 8
        max_weight_limit = max_weight_query.value / U16_MAX if max_weight_query else 0.1
        
        logger.debug(f"Chain params - min_allowed: {min_allowed_weights}, max_limit: {max_weight_limit}")
        
        # Get non-zero weights
        non_zero_idx = np.argwhere(weights > 0).squeeze()
        non_zero_idx = np.atleast_1d(non_zero_idx)
        non_zero_weights = weights[non_zero_idx]
        non_zero_uids = uids[non_zero_idx]
        
        logger.debug(f"Non-zero weights: {len(non_zero_weights)}/{len(weights)}")
        
        # Check if we have enough non-zero weights
        if non_zero_weights.size == 0:
            logger.warning("No non-zero weights, cannot set")
            return [], []
        
        if non_zero_weights.size < min_allowed_weights:
            logger.warning(
                f"Only {non_zero_weights.size} non-zero weights, "
                f"but chain requires {min_allowed_weights}"
            )
            return [], []
        
        # Normalize with max weight limit
        processed_weights = self._normalize_max_weight(
            non_zero_weights,
            limit=max_weight_limit
        )
        processed_uids = non_zero_uids
        
        logger.debug(f"Processed weights sum: {processed_weights.sum()}")
        
        # Convert to uint16
        uint_uids, uint_weights = self._convert_weights_and_uids_for_emit(
            uids=processed_uids,
            weights=processed_weights
        )
        
        # Convert back to float for chain submission
        node_weights = [float(w) / float(U16_MAX) for w in uint_weights]
        node_ids = [int(uid) for uid in uint_uids]
        
        return node_ids, node_weights
    
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
        force: bool = False,
    ) -> WeightSetResult:
        """
        Set weights on blockchain with full chain parameter processing.
        
        Args:
            uids: List of miner UIDs
            weights: List of corresponding raw weights
            force: Force weight set even if within min_interval
            
        Returns:
            WeightSetResult with operation details
        """
        # Check rate limit
        if not force:
            can_set, remaining = self.can_set_weights()
            if not can_set:
                logger.warning(f"Rate limited. Wait {remaining:.1f}s")
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message=f"Rate limited. Wait {remaining:.1f}s",
                )
        
        logger.info(f"Setting weights for {len(uids)} miners (raw sum: {sum(weights):.4f})")
        
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
                        f"Weights set successfully on attempt {attempt} "
                        f"(total: {self.total_sets}, failed: {self.failed_sets})"
                    )
                    return result
                else:
                    last_error = result.error_message
                    logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {last_error}")
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt}/{self.max_retries} error: {e}")
            
            if attempt < self.max_retries:
                logger.info(f"Retrying in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)
        
        # All retries failed
        self.failed_sets += 1
        logger.error(f"Failed after {self.max_retries} attempts: {last_error}")
        return WeightSetResult(
            success=False,
            uids=uids,
            weights=weights,
            error_message=f"Failed after {self.max_retries} attempts: {last_error}",
            attempts=self.max_retries,
        )
    
    async def _set_weights_with_timeout(
        self,
        substrate: SubstrateInterface,
        node_ids: List[int],
        node_weights: List[float],
        validator_node_id: int,
        version_key: int,
    ) -> bool:
        """
        Set weights with timeout protection.
        
        Args:
            substrate: Substrate interface
            node_ids: List of node IDs
            node_weights: List of node weights
            validator_node_id: Validator's node ID
            version_key: Version key from chain
            
        Returns:
            Success boolean
        """
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    weights.set_node_weights,
                    substrate,
                    self.keypair,
                    node_ids,
                    node_weights,
                    self.netuid,
                    validator_node_id,
                    version_key,
                    True,  # wait_for_inclusion
                    True,  # wait_for_finalization
                ),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Weight setting timed out after {self.timeout}s")
            return False
        except Exception as e:
            logger.error(f"Error in set_node_weights: {e}")
            return False
    
    async def _do_set_weights(
        self,
        uids: List[int],
        weights: List[float],
    ) -> WeightSetResult:
        """
        Process and set weights on chain using fiber.
        
        Args:
            uids: List of UIDs
            weights: List of raw weights
            
        Returns:
            WeightSetResult
        """
        try:
            # Get substrate connection
            substrate = self._get_substrate()
            
            # Query validator node ID
            node_id_query = substrate.query(
                "SubtensorModule",
                "Uids",
                [self.netuid, self.keypair.ss58_address]
            )
            
            if not node_id_query:
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message="Failed to query validator node ID"
                )
            
            validator_node_id = node_id_query.value
            logger.debug(f"Validator node ID: {validator_node_id}")
            
            # Query version key
            version_key_query = substrate.query(
                "SubtensorModule",
                "WeightsVersionKey",
                [self.netuid]
            )
            
            version_key = version_key_query.value if version_key_query else 0
            logger.debug(f"Version key: {version_key}")
            
            # Convert to numpy arrays
            uids_array = np.array(uids, dtype=np.int64)
            weights_array = np.array(weights, dtype=np.float32)
            
            # Process weights according to chain parameters
            node_ids, node_weights = self._process_weights_for_netuid(
                uids=uids_array,
                weights=weights_array,
                substrate=substrate
            )
            
            if not node_ids:
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message="No valid weights after processing"
                )
            
            logger.info(f"Submitting {len(node_ids)} weights to chain: {list(zip(node_ids, node_weights))[:5]}...")
            
            # Set weights with timeout
            success = await self._set_weights_with_timeout(
                substrate=substrate,
                node_ids=node_ids,
                node_weights=node_weights,
                validator_node_id=validator_node_id,
                version_key=version_key
            )
            
            if success:
                return WeightSetResult(
                    success=True,
                    uids=node_ids,
                    weights=node_weights,
                )
            else:
                return WeightSetResult(
                    success=False,
                    uids=uids,
                    weights=weights,
                    error_message="Chain weight setting failed"
                )
        
        except Exception as e:
            logger.error(f"Error in _do_set_weights: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return WeightSetResult(
                success=False,
                uids=uids,
                weights=weights,
                error_message=str(e)
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
            "wallet": f"{self.wallet_name}/{self.hotkey_name}",
        }