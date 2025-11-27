"""
Scoring Cache Service

Proactive cache management for /scoring endpoint with full refresh strategy.
Simplified design: always performs full refresh every 5 minutes.
"""

import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from affine.core.setup import logger


def _format_timestamp(ts_ms: int) -> str:
    """Convert millisecond timestamp to human-readable format."""
    if ts_ms == 0:
        return "0 (epoch)"
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000.0)
        return f"{ts_ms} ({dt.strftime('%Y-%m-%d %H:%M:%S')})"
    except (ValueError, OSError):
        return f"{ts_ms} (invalid)"


class CacheState(Enum):
    """Cache state machine."""
    EMPTY = "empty"
    WARMING = "warming"
    READY = "ready"
    REFRESHING = "refreshing"


@dataclass
class CacheConfig:
    """Cache configuration."""
    refresh_interval: int = 300  # 5 minutes


class ScoringCacheManager:
    """Manages scoring data cache with full refresh strategy."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Cache data
        self._data: Dict[str, Any] = {}
        self._state = CacheState.EMPTY
        self._lock = asyncio.Lock()
        
        # Timestamp
        self._updated_at = 0
        
        # Background task
        self._refresh_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> CacheState:
        return self._state
    
    async def warmup(self) -> None:
        """Warm up cache on startup."""
        logger.info("Warming up scoring cache...")
        
        async with self._lock:
            self._state = CacheState.WARMING
            try:
                await self._full_refresh()
                self._state = CacheState.READY
                self._updated_at = int(time.time())
                logger.info(f"Cache warmed up ({len(self._data)} miners)")
            except Exception as e:
                logger.error(f"Failed to warm up cache: {e}", exc_info=True)
                self._state = CacheState.EMPTY
    
    async def get_data(self) -> Dict[str, Any]:
        """Get cached scoring data with fallback logic.
        
        Non-blocking: Returns cached data immediately when READY or REFRESHING.
        Blocking: Waits for initial warmup when EMPTY or WARMING.
        """
        # Fast path: return cache if ready or refreshing
        if self._state in [CacheState.READY, CacheState.REFRESHING]:
            return self._data
        
        # Slow path: cache empty or warming
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
            # Timeout - but check if data exists
            if self._data:
                logger.warning("Cache warming timeout but returning existing data")
                return self._data
            raise TimeoutError("Cache warming timeout")
        
        # Fallback: return any available data
        if self._data:
            logger.warning(f"Returning stale cache (state={self._state})")
            return self._data
        
        raise RuntimeError(f"No cache data available (state={self._state})")
    
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
        """Execute full refresh."""
        start_time = time.time()
        logger.info("Full refresh started")
        
        from affine.database.dao.system_config import SystemConfigDAO
        from affine.database.dao.miners import MinersDAO
        from affine.database.dao.sample_results import SampleResultsDAO
        
        system_config_dao = SystemConfigDAO()
        miners_dao = MinersDAO()
        sample_dao = SampleResultsDAO()
        
        # Get config
        scoring_envs = await system_config_dao.get_active_environments()
        if not scoring_envs:
            self._data = {}
            logger.info("Full refresh completed: no active environments")
            return
        
        env_ranges_dict = await system_config_dao.get_env_task_ranges()
        valid_miners = await miners_dao.get_valid_miners()
        if not valid_miners:
            self._data = {}
            logger.info("Full refresh completed: no valid miners")
            return
        
        # Build query params
        env_ranges = {
            env: tuple(env_ranges_dict[env]['scoring_range'])
            for env in scoring_envs
            if env in env_ranges_dict
        }
        
        miners_list = [
            {'hotkey': m['hotkey'], 'revision': m['revision']}
            for m in valid_miners
        ]
        
        # Batch query
        samples_data = await sample_dao.get_scoring_samples_batch(
            miners=miners_list,
            env_ranges=env_ranges
        )
        
        # Assemble result
        result = self._assemble_result(valid_miners, scoring_envs, env_ranges, samples_data)
        
        # Calculate statistics
        total_samples = 0
        max_timestamp = 0
        for miner_data in result.values():
            for env_data in miner_data.get('env', {}).values():
                total_samples += env_data.get('completed_count', 0)
                for sample in env_data.get('samples', []):
                    max_timestamp = max(max_timestamp, sample.get('timestamp', 0))
        
        # Update cache
        self._data = result
        
        # Log statistics
        elapsed = time.time() - start_time
        combo_count = len(valid_miners) * len(scoring_envs)
        logger.info(
            f"Full refresh completed: {len(result)} miners, "
            f"{combo_count} miner×env combinations, "
            f"{total_samples} total samples, "
            f"max_ts={_format_timestamp(max_timestamp)}, "
            f"elapsed={elapsed:.2f}s"
        )
    
    def _assemble_result(
        self,
        miners: list,
        envs: list,
        env_ranges: dict,
        samples_data: dict
    ) -> Dict[str, Any]:
        """Assemble scoring result from query data."""
        result = {}
        
        for miner in miners:
            uid = miner['uid']
            hotkey = miner['hotkey']
            revision = miner['revision']
            key = f"{hotkey}#{revision}"
            
            miner_samples = samples_data.get(key, {})
            
            miner_entry = {
                'hotkey': hotkey,
                'model_revision': revision,
                'model_repo': miner.get('model'),
                'first_block': miner.get('first_block'),
                'env': {}
            }
            
            for env in envs:
                env_samples = miner_samples.get(env, [])
                start_id, end_id = env_ranges.get(env, (0, 0))
                
                if start_id >= end_id:
                    continue
                
                samples_list = [
                    {
                        'task_id': int(s['task_id']),
                        'score': s['score'],
                        'task_uuid': s['timestamp'],
                        'timestamp': s['timestamp'],
                    }
                    for s in env_samples
                ]
                
                expected_count = end_id - start_id
                completed_count = len(samples_list)
                completeness = completed_count / expected_count if expected_count > 0 else 0.0
                
                completed_task_ids = {s['task_id'] for s in samples_list}
                all_task_ids = set(range(start_id, end_id))
                missing_task_ids = sorted(list(all_task_ids - completed_task_ids))[:100]
                
                miner_entry['env'][env] = {
                    'samples': samples_list,
                    'total_count': expected_count,
                    'completed_count': completed_count,
                    'missing_task_ids': missing_task_ids,
                    'completeness': round(completeness, 4)
                }
            
            result[str(uid)] = miner_entry
        
        return result


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
    """Get cached scoring data."""
    return await _cache_manager.get_data()