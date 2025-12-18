"""
Scoring Cache Service

Proactive cache management for /scoring endpoint with full refresh strategy.
Simplified design: always performs full refresh every 5 minutes.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from affine.core.setup import logger


class CacheState(Enum):
    """Cache state machine."""
    EMPTY = "empty"
    WARMING = "warming"
    READY = "ready"
    REFRESHING = "refreshing"


@dataclass
class CacheConfig:
    """Cache configuration."""
    refresh_interval: int = 600  # 5 minutes


class ScoringCacheManager:
    """Manages scoring data cache with full refresh strategy."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Cache data for scoring (enabled_for_scoring=true environments only)
        self._data: Dict[str, Any] = {}
        self._state = CacheState.EMPTY
        self._lock = asyncio.Lock()
        
        # Timestamp for cache
        self._updated_at = 0
        
        # Background task
        self._refresh_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> CacheState:
        return self._state
    
    async def warmup(self) -> None:
        """Warm up cache on startup."""
        logger.info("Warming up scoring cache (enabled_for_scoring environments only)...")
        
        async with self._lock:
            self._state = CacheState.WARMING
            try:
                await self._full_refresh()
                self._state = CacheState.READY
                self._updated_at = int(time.time())
                logger.info(f"Cache warmed up: {len(self._data)} miners")
            except Exception as e:
                logger.error(f"Failed to warm up cache: {e}", exc_info=True)
                self._state = CacheState.EMPTY
    
    async def get_data(self) -> Dict[str, Any]:
        """Get cached scoring data with fallback logic.
        
        Non-blocking: Returns cached data immediately when READY or REFRESHING.
        Blocking: Waits for initial warmup when EMPTY or WARMING.
        """
        # Fast path: return cache if ready or refreshing (data can be empty dict)
        if self._state in [CacheState.READY, CacheState.REFRESHING]:
            return self._data
        
        # Slow path: cache not initialized yet
        if self._state == CacheState.EMPTY:
            async with self._lock:
                # Double check after acquiring lock
                if self._state == CacheState.EMPTY:
                    logger.warning("Cache miss - computing synchronously")
                    self._state = CacheState.WARMING
                    try:
                        await self._full_refresh()
                        self._state = CacheState.READY
                        self._updated_at = int(time.time())
                        return self._data
                    except Exception as e:
                        self._state = CacheState.EMPTY
                        raise RuntimeError(f"Failed to compute scoring data: {e}") from e
        
        # Warming in progress - wait and recheck
        if self._state == CacheState.WARMING:
            for _ in range(60):
                await asyncio.sleep(1)
                # Recheck state - may have changed to READY
                if self._state == CacheState.READY:
                    return self._data
            # Timeout - return whatever we have
            logger.warning("Cache warming timeout, returning current data")
            return self._data
        
        # Fallback: return any available data (should not reach here)
        logger.warning(f"Returning cache in unexpected state (state={self._state})")
        return self._data
    
    async def start_refresh_loop(self) -> None:
        """Start background refresh loop."""
        self._refresh_task = asyncio.create_task(self._refresh_loop())
    
    async def stop_refresh_loop(self) -> None:
        """Stop background refresh loop."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
    
    async def _refresh_loop(self) -> None:
        """Background refresh loop with full refresh strategy."""
        while True:
            try:
                await asyncio.sleep(self.config.refresh_interval)
                
                # Set refreshing state (non-blocking for API access)
                async with self._lock:
                    if self._state == CacheState.READY:
                        self._state = CacheState.REFRESHING
                
                # Always perform full refresh
                await self._full_refresh()
                
                # Mark ready
                async with self._lock:
                    self._state = CacheState.READY
                    self._updated_at = int(time.time())
                
            except asyncio.CancelledError:
                logger.info("Cache refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Cache refresh failed: {e}", exc_info=True)
                async with self._lock:
                    if self._state == CacheState.REFRESHING:
                        self._state = CacheState.READY
    
    async def _full_refresh(self) -> None:
        """Execute full refresh with NEW incremental update strategy.
        
        NEW DESIGN:
        - Uses sampling_list from sampling_config if available
        - Query by PK+SK for each miner+env+taskid (no range queries)
        - Incremental updates based on task ID differences
        - Detects miner changes and removes invalid cache
        """
        start_time = time.time()
        logger.info("Full refresh started (incremental update strategy)")
        
        from affine.database.dao.system_config import SystemConfigDAO
        from affine.database.dao.miners import MinersDAO
        from affine.database.dao.sample_results import SampleResultsDAO
        
        system_config_dao = SystemConfigDAO()
        miners_dao = MinersDAO()
        sample_dao = SampleResultsDAO()
        
        # 1. Get current valid miners
        valid_miners = await miners_dao.get_valid_miners()
        current_miner_keys = {
            (m['hotkey'], m['revision']) for m in valid_miners
        }
        
        # 2. Detect miner changes
        previous_miner_keys = getattr(self, '_previous_miner_keys', set())
        removed_miners = previous_miner_keys - current_miner_keys
        
        # 3. Remove invalid miner cache
        if removed_miners:
            for hotkey, revision in removed_miners:
                key = f"{hotkey}#{revision}"
                if key in self._data:
                    del self._data[key]
                    logger.info(f"Removed cache for invalid miner {hotkey[:8]}...#{revision[:8]}...")
        
        # 4. Get environment configurations
        environments = await system_config_dao.get_param_value('environments', {})
        
        if not environments:
            self._data = {}
            logger.info("Full refresh completed: no environments configured")
            return
        
        if not valid_miners:
            self._data = {}
            logger.info("Full refresh completed: no valid miners")
            return
        
        # 5. Update cache for each miner+env combination
        for miner in valid_miners:
            hotkey = miner['hotkey']
            revision = miner['revision']
            key = f"{hotkey}#{revision}"
            uid = miner['uid']
            
            # Initialize miner entry
            if key not in self._data:
                self._data[key] = {
                    'hotkey': hotkey,
                    'model_revision': revision,
                    'model_repo': miner.get('model'),
                    'first_block': miner.get('first_block'),
                    'env': {}
                }
            
            # For backward compatibility, also use UID as key
            self._data[str(uid)] = self._data[key]
            
            # Only update cache for enabled_for_scoring environments
            for env_name, env_config in environments.items():
                if env_config.get('enabled_for_scoring'):
                    await self._update_miner_env_cache(
                        sample_dao=sample_dao,
                        miner_key=key,
                        miner_info=miner,
                        env=env_name,
                        env_config=env_config
                    )
        
        # 6. Update miner tracking
        self._previous_miner_keys = current_miner_keys
        
        # Log statistics
        elapsed = time.time() - start_time
        combo_count = len(valid_miners) * len(environments)
        logger.info(
            f"Full refresh completed: {len(valid_miners)} miners, "
            f"{len(environments)} environments, "
            f"{combo_count} miner×env combinations, "
            f"elapsed={elapsed:.2f}s"
        )
    
    async def _update_miner_env_cache(
        self,
        sample_dao,
        miner_key: str,
        miner_info: dict,
        env: str,
        env_config: dict
    ):
        """Incremental update for single miner+env cache.
        
        Strategy:
        1. Get target task IDs from sampling_config or ranges
        2. Compare with current cache
        3. Query new task IDs
        4. Remove obsolete task IDs
        """
        from affine.core.sampling_list import get_task_id_set_from_config
        
        hotkey = miner_info['hotkey']
        revision = miner_info['revision']
        
        # 1. Get target task IDs (prioritize sampling_list, fallback to ranges)
        target_task_ids = get_task_id_set_from_config(env_config)
        
        if not target_task_ids:
            return
        
        # 2. Get current cache
        current_cache = self._data[miner_key]['env'].get(env, {})
        current_samples = current_cache.get('samples', [])
        current_task_ids = {s['task_id'] for s in current_samples}
        
        # 3. Calculate differences
        added_task_ids = target_task_ids - current_task_ids
        removed_task_ids = current_task_ids - target_task_ids
        
        # 4. Query new task IDs
        new_samples = []
        for task_id in added_task_ids:
            try:
                sample = await sample_dao.get_sample_by_task_id(
                    miner_hotkey=hotkey,
                    model_revision=revision,
                    env=env,
                    task_id=str(task_id),
                    include_extra=False
                )
                if sample:
                    new_samples.append({
                        'task_id': task_id,
                        'score': sample['score'],
                        'task_uuid': sample['timestamp'],
                        'timestamp': sample['timestamp']
                    })
            except Exception as e:
                logger.debug(f"Failed to query sample {env}/{task_id} for {hotkey[:8]}...: {e}")
        
        # 5. Remove obsolete task IDs
        updated_samples = [
            s for s in current_samples
            if s['task_id'] not in removed_task_ids
        ]
        
        # 6. Merge new samples
        updated_samples.extend(new_samples)
        
        # 7. Calculate statistics
        expected_count = len(target_task_ids)
        completed_count = len(updated_samples)
        completeness = completed_count / expected_count if expected_count > 0 else 0.0
        
        completed_task_ids = {s['task_id'] for s in updated_samples}
        missing_task_ids = sorted(list(target_task_ids - completed_task_ids))[:100]
        
        # 8. Update cache
        self._data[miner_key]['env'][env] = {
            'samples': updated_samples,
            'total_count': expected_count,
            'completed_count': completed_count,
            'missing_task_ids': missing_task_ids,
            'completeness': round(completeness, 4)
        }
    


# Global cache manager instance
_cache_manager = ScoringCacheManager()


# Public API
async def warmup_cache() -> None:
    """Warm up cache on startup."""
    await _cache_manager.warmup()


async def refresh_cache_loop() -> None:
    """Start background refresh loop."""
    await _cache_manager.start_refresh_loop()


async def get_cached_data() -> Dict[str, Any]:
    """Get cached scoring data (enabled_for_scoring environments only)."""
    return await _cache_manager.get_data()