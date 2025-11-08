"""
Scheduler Initializer - Load historical sampling data to initialize MinerSamplers.

This module provides a simple, decoupled way to load historical sampling statistics
from R2 storage and Summary files, allowing the scheduler to resume with accurate state
after restart.

Design goals:
1. Simple implementation - reuse existing storage.dataset() and storage.load_summary()
2. Decoupled architecture - easy to replace with backend API in the future
3. Efficient loading - load only recent 600 blocks (~2 hours) for last sample times
4. Accurate rate adjustment - use Summary data to determine if 3x sampling is needed
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict

from affine.storage import dataset, load_summary
from affine.setup import logger


class SchedulerInitializer:
    """Initialize scheduler with historical sampling data"""
    
    def __init__(self):
        """Initialize the scheduler initializer"""
        pass
    
    async def load_init_data(self, current_block: int) -> Dict[int, Dict[str, Any]]:
        """Load initialization data for all miners.
        
        This method loads:
        1. Summary data - to determine total sample counts (for 3x rate multiplier decision)
        2. Recent 600 blocks (~2 hours) - to get last sample timestamps per env (for state display)
        
        Args:
            current_block: Current blockchain block number
            
        Returns:
            Dictionary mapping UID to initialization data:
            {
                uid: {
                    'hotkey': str,
                    'total_samples': int,              # From Summary (for rate multiplier)
                    'env_last_sample_time': {          # From recent blocks (for state display)
                        'env_name': timestamp
                    }
                }
            }
        """
        logger.info("[Initializer] Loading historical data for scheduler initialization")
        
        # Initialize result structure
        init_data: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            'hotkey': None,
            'total_samples': 0,
            'env_last_sample_time': {}
        })
        
        # Step 1: Load Summary to get total sample counts
        try:
            summary_data = await self._load_summary_data()
            for uid, data in summary_data.items():
                init_data[uid]['hotkey'] = data['hotkey']
                init_data[uid]['total_samples'] = data['total_samples']
            
            logger.info(f"[Initializer] Loaded Summary data for {len(summary_data)} miners")
        except Exception as e:
            logger.warning(f"[Initializer] Failed to load Summary, continuing without it: {e}")
        
        # Step 2: Load recent 600 blocks to get last sample times
        try:
            env_timestamps = await self._load_recent_samples(current_block)
            for uid, env_times in env_timestamps.items():
                init_data[uid]['env_last_sample_time'] = env_times
            
            logger.info(f"[Initializer] Loaded recent samples for {len(env_timestamps)} miners")
        except Exception as e:
            logger.warning(f"[Initializer] Failed to load recent samples, continuing without them: {e}")
        
        # Convert defaultdict to regular dict
        result = dict(init_data)
        logger.info(f"[Initializer] Initialization complete: {len(result)} miners with historical data")
        
        return result
    
    async def _load_summary_data(self) -> Dict[int, Dict[str, Any]]:
        """Load Summary file and extract total sample counts per miner.
        
        Returns:
            Dictionary mapping UID to summary data:
            {
                uid: {
                    'hotkey': str,
                    'total_samples': int  # Sum of all environment counts
                }
            }
        """
        summary = await load_summary()
        
        # Extract miners data from summary
        miners_data = summary.get('data', {}).get('miners', {})
        
        result = {}
        for hotkey, miner_info in miners_data.items():
            uid = miner_info.get('uid')
            if uid is None:
                continue
            
            # Sum up all environment counts
            environments = miner_info.get('environments', {})
            total_count = sum(
                env_data.get('count', 0) 
                for env_data in environments.values()
            )
            
            result[uid] = {
                'hotkey': hotkey,
                'total_samples': total_count
            }
        
        return result
    
    async def _load_recent_samples(
        self, 
        current_block: int,
        blocks: int = 600,
        time_window: int = 3600
    ) -> Dict[int, Dict[str, float]]:
        """Load recent 600 blocks and extract last sample time per environment.
        
        Args:
            current_block: Current blockchain block number
            blocks: Number of blocks to load (default: 600 ≈ 2 hours)
            time_window: Time window in seconds to filter samples (default: 3600 = 1 hour)
            
        Returns:
            Dictionary mapping UID to environment timestamps:
            {
                uid: {
                    'env_name': timestamp  # Latest timestamp for this env
                }
            }
        """
        # Calculate time threshold (only count samples within last hour)
        one_hour_ago = time.time() - time_window
        
        # Track latest timestamp per (uid, env)
        env_timestamps: Dict[int, Dict[str, float]] = defaultdict(dict)
        
        # Load recent blocks using existing dataset function
        count = 0
        async for result in dataset(tail=blocks, compact=True):
            count += 1
            
            # Skip samples older than 1 hour
            if result.timestamp < one_hour_ago:
                continue
            
            uid = result.miner.uid
            env = result.env
            timestamp = result.timestamp
            
            # Track the latest timestamp for each (uid, env) pair
            if env not in env_timestamps[uid]:
                env_timestamps[uid][env] = timestamp
            else:
                env_timestamps[uid][env] = max(env_timestamps[uid][env], timestamp)
        
        logger.info(f"[Initializer] Processed {count} samples from last {blocks} blocks")
        
        # Convert defaultdict to regular dict
        return dict(env_timestamps)