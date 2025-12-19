"""
Sampling List Manager

Manages dynamic sampling list generation and rotation logic.
"""

import random
from typing import List, Tuple, Set, Dict, Any

from affine.core.setup import logger


def get_task_id_set_from_config(env_config: Dict[str, Any]) -> Set[int]:
    """Get task ID set from environment configuration.
    
    Prioritizes sampling_list from sampling_config.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Set of task IDs
    """
    sampling_config = env_config.get('sampling_config', {})
    
    # Use sampling_list
    sampling_list = sampling_config.get('sampling_list')
    if sampling_list:
        return set(sampling_list)
    
    # If no sampling_list, return empty set (should not happen in normal operation)
    logger.warning(
        f"No sampling_list found in sampling_config, returning empty set. "
        f"Config: {sampling_config}"
    )
    return set()


class SamplingListManager:
    """Sampling list manager for dynamic task rotation."""
    
    async def initialize_sampling_list(
        self,
        env: str,
        initial_range: List[List[int]],
        sampling_size: int
    ) -> List[int]:
        """Initialize sampling list from initial range.
        
        Simply expands initial_range to task ID set and randomly samples.
        
        Args:
            env: Environment name
            initial_range: Initial range in [[start, end], ...] format
            sampling_size: Target sampling list size
            
        Returns:
            Initialized task ID list (may be smaller than sampling_size if insufficient IDs)
        """
        from affine.database.dao.system_config import ranges_to_task_id_set
        
        # Expand initial_range to task ID set
        initial_ids = ranges_to_task_id_set(initial_range)
        
        # Randomly sample up to sampling_size
        actual_size = min(len(initial_ids), sampling_size)
        sampling_list = random.sample(list(initial_ids), actual_size)
        
        logger.info(
            f"Initialized sampling list for {env}: "
            f"sampled {actual_size} from {len(initial_ids)} initial IDs (target={sampling_size})"
        )
        
        return sorted(sampling_list)
    
    async def rotate_sampling_list(
        self,
        env: str,
        current_list: List[int],
        dataset_range: List[List[int]],
        sampling_count: int,
        rotation_count: int
    ) -> Tuple[List[int], List[int], List[int]]:
        """Rotate sampling list while maintaining sampling_count size.
        
        Clean set-based logic:
        1. Convert dataset_range to set (supports non-contiguous ranges)
        2. Calculate available = dataset - current
        3. Determine removal/addition counts based on current vs target size
        4. Skip if insufficient IDs after deduplication
        
        Args:
            env: Environment name
            current_list: Current sampling list
            dataset_range: Dataset range in [[start, end], ...] format (supports multiple ranges)
            sampling_count: Target sampling list size
            rotation_count: Number of IDs to rotate per cycle
            
        Returns:
            (new_list, removed_ids, added_ids)
        """
        from affine.database.dao.system_config import ranges_to_task_id_set
        
        if rotation_count < 0:
            logger.warning(f"Invalid rotation_count for {env}: {rotation_count}")
            return current_list, [], []
        
        # rotation_count=0 is valid: only adjust size, no rotation
        
        # Convert to sets
        dataset_set = ranges_to_task_id_set(dataset_range)
        current_set = set(current_list)
        available_set = dataset_set - current_set
        
        # Safety check: Skip if would use > 80% of dataset
        if sampling_count + rotation_count > len(dataset_set) * 0.8:
            logger.warning(
                f"Skipping rotation for {env}: safety check failed - "
                f"sampling_count ({sampling_count}) + rotation_count ({rotation_count}) "
                f"> 80% of dataset ({len(dataset_set) * 0.8:.0f})"
            )
            return current_list, [], []
        
        current_size = len(current_list)
        
        # Determine removal and addition counts
        if current_size < sampling_count:
            # Fill mode: Only add to reach target
            to_remove = 0
            to_add = sampling_count - current_size
        elif current_size > sampling_count:
            # Shrink+Rotate mode: Remove surplus + rotation_count, add rotation_count
            surplus = current_size - sampling_count
            to_remove = surplus + rotation_count
            to_add = rotation_count
        else:
            # Standard rotation: Remove N, add N
            to_remove = rotation_count
            to_add = rotation_count
        
        # Skip if not enough available IDs
        if len(available_set) < to_add:
            logger.warning(
                f"Skipping rotation for {env}: insufficient available IDs - "
                f"need={to_add}, available={len(available_set)}"
            )
            return current_list, [], []
        
        # Execute removal
        if to_remove > 0:
            to_remove = min(to_remove, current_size)
            removed_ids = random.sample(current_list, to_remove)
            remaining_set = current_set - set(removed_ids)
        else:
            removed_ids = []
            remaining_set = current_set
        
        # Execute addition (recalculate available after removal)
        available_for_add = dataset_set - remaining_set
        added_ids = random.sample(list(available_for_add), to_add)
        
        # Merge and sort
        new_set = remaining_set | set(added_ids)
        new_list = sorted(list(new_set))
        
        logger.info(
            f"Rotated {env}: removed={len(removed_ids)}, added={len(added_ids)}, "
            f"size: {current_size} -> {len(new_list)} (target={sampling_count})"
        )
        
        return new_list, removed_ids, added_ids