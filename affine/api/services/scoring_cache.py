"""
Scoring Cache Service

Proactive cache management for /scoring endpoint with incremental updates.
"""

import time
import asyncio
import copy
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
    refresh_interval: int = 300  # 5 minutes (incremental check)
    miner_count_diff_threshold: int = 5


class ScoringCacheManager:
    """Manages scoring data cache with incremental updates."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Cache data
        self._data: Dict[str, Any] = {}
        self._state = CacheState.EMPTY
        self._lock = asyncio.Lock()
        
        # Timestamps
        self._updated_at = 0
        self._last_sample_ts = 0
        
        # Metadata for change detection
        self._cached_envs = []
        
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
                logger.info(f"Cache warmed up ({len(self._data)} miners, max_ts={_format_timestamp(self._last_sample_ts)})")
            except Exception as e:
                logger.error(f"Failed to warm up cache: {e}", exc_info=True)
                self._state = CacheState.EMPTY
    
    async def get_data(self) -> Dict[str, Any]:
        """Get cached scoring data with fallback logic."""
        # Fast path: return cache if ready or refreshing
        # Note: Background refresh ensures data freshness, no need for TTL check
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
    
    async def _fetch_current_config(self):
        """Fetch current system configuration.
        
        Returns:
            Tuple of (envs, env_ranges, valid_miners, current_miners)
        """
        from affine.database.dao.system_config import SystemConfigDAO
        from affine.database.dao.miners import MinersDAO
        
        system_config_dao = SystemConfigDAO()
        miners_dao = MinersDAO()
        
        current_envs = await system_config_dao.get_active_environments()
        env_ranges_dict = await system_config_dao.get_env_task_ranges()
        current_env_ranges = {
            env: tuple(env_ranges_dict[env]['scoring_range'])
            for env in current_envs
            if env in env_ranges_dict
        }
        valid_miners = await miners_dao.get_valid_miners()
        current_miners = {str(m['uid']): {'hotkey': m['hotkey'], 'revision': m['revision']} for m in valid_miners}
        
        return current_envs, current_env_ranges, valid_miners, current_miners
    
    async def _refresh_loop(self) -> None:
        """Background refresh loop with incremental strategy."""
        while True:
            try:
                await asyncio.sleep(self.config.refresh_interval)
                
                # Fetch current configuration once
                current_envs, current_env_ranges, valid_miners, current_miners = await self._fetch_current_config()
                
                # Set refreshing state
                async with self._lock:
                    if self._state == CacheState.READY:
                        self._state = CacheState.REFRESHING
                
                # Choose strategy based on config
                if self._should_full_refresh_sync(current_miners, current_envs, current_env_ranges):
                    await self._full_refresh()
                else:
                    await self._incremental_refresh_with_config(
                        current_envs, current_env_ranges, valid_miners, current_miners
                    )
                
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
    
    def _should_full_refresh_sync(
        self,
        current_miners: dict,
        current_envs: list,
        current_env_ranges: dict
    ) -> bool:
        """Determine if full refresh is needed (synchronous).
        
        Strategy: Detect major configuration changes requiring full rebuild.
        
        Args:
            current_miners: Current miner dict from config
            current_envs: Current active environments
            current_env_ranges: Current env task ranges
        
        Returns:
            True if full refresh needed
        """
        # First run - always full refresh
        if not self._cached_miners:
            logger.info("Full refresh: first run")
            return True
        
        # Check if too many miner changes
        if abs(len(current_miners) - len(self._cached_miners)) > self.config.miner_count_diff_threshold:
            logger.info(f"Full refresh: miner count changed significantly ({len(self._cached_miners)} → {len(current_miners)})")
            return True
        
        # Check if environment list changed
        if set(current_envs) != set(self._cached_envs):
            logger.info(f"Full refresh: environment list changed ({self._cached_envs} → {current_envs})")
            return True
        
        # Check if any task range changed
        for env in current_envs:
            old_range = self._cached_env_ranges.get(env)
            new_range = current_env_ranges.get(env)
            if old_range != new_range:
                logger.info(f"Full refresh: task range changed for {env} ({old_range} → {new_range})")
                return True
        
        return False
    
    async def _full_refresh(self) -> None:
        """Execute full refresh."""
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
            return
        
        env_ranges_dict = await system_config_dao.get_env_task_ranges()
        valid_miners = await miners_dao.get_valid_miners()
        if not valid_miners:
            self._data = {}
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
        
        # Calculate total samples loaded
        total_samples = 0
        for miner_data in result.values():
            for env_data in miner_data.get('env', {}).values():
                total_samples += env_data.get('completed_count', 0)
        
        # Update cache metadata
        self._data = result
        self._cached_envs = scoring_envs
        self._cached_env_ranges = env_ranges
        self._cached_miners = {str(m['uid']): {'hotkey': m['hotkey'], 'revision': m['revision']} for m in valid_miners}
        
        # Extract global max timestamp from all samples
        all_timestamps = []
        for miner_samples in samples_data.values():
            for env_samples in miner_samples.values():
                all_timestamps.extend(s['timestamp'] for s in env_samples)
        
        self._last_sample_ts = max(all_timestamps, default=0)
        
        # Calculate miner×env combinations count
        combo_count = len(valid_miners) * len(scoring_envs)
        logger.info(f"Full refresh completed: {len(result)} miners, {combo_count} miner×env combinations, {total_samples} total samples")
    
    async def _incremental_refresh_with_config(
        self,
        current_envs: list,
        current_env_ranges: dict,
        valid_miners: list,
        current_miners: dict
    ) -> None:
        """Execute fine-grained incremental refresh with pre-fetched config.
        
        Strategy:
        1. Detect config changes (new miners, new envs, range changes, removed envs)
        2. For each miner×env:
           - New combination → full scan
           - Range changed → full scan for that env
           - Miner changed → full scan for that miner
           - Otherwise → incremental scan since last_scanned_ts
        3. Remove deleted envs from cache
        
        Args:
            current_envs: Current active environments
            current_env_ranges: Current env task ranges
            valid_miners: Current valid miners list
            current_miners: Current miners dict
        """
        logger.info("Fine-grained incremental refresh started")
        
        from affine.database.dao.sample_results import SampleResultsDAO
        sample_dao = SampleResultsDAO()
        
        # Detect miner changes
        added_miners = {uid: info for uid, info in current_miners.items() if uid not in self._cached_miners}
        changed_miners = {
            uid: info for uid, info in current_miners.items()
            if uid in self._cached_miners and (
                info['hotkey'] != self._cached_miners[uid]['hotkey'] or
                info['revision'] != self._cached_miners[uid]['revision']
            )
        }
        removed_miners = {uid for uid in self._cached_miners if uid not in current_miners}
        
        logger.info(f"Changes: +{len(added_miners)} miners, Δ{len(changed_miners)} miners, -{len(removed_miners)} miners")
        
        # Build scan plan
        full_scan_combos = []  # (hotkey, revision, env) requiring full scan
        incr_scan_combos = []  # (hotkey, revision, env, since_ts) requiring incremental scan
        
        for miner in valid_miners:
            uid = str(miner['uid'])
            hotkey, revision = miner['hotkey'], miner['revision']
            
            is_new_miner = uid in added_miners
            is_changed_miner = uid in changed_miners
            
            for env in current_envs:
                # Decision: full vs incremental
                if is_new_miner or is_changed_miner:
                    # Full scan needed
                    full_scan_combos.append((hotkey, revision, env))
                else:
                    # Incremental scan (using global last_sample_ts)
                    incr_scan_combos.append((hotkey, revision, env))
        
        logger.info(f"Scan plan: {len(full_scan_combos)} full, {len(incr_scan_combos)} incremental")
        
        # Step 3: Execute scans
        new_samples = {}
        new_timestamps = []  # Collect all new timestamps
        
        if full_scan_combos:
            # Extract unique (hotkey, revision) pairs from full_scan_combos
            unique_miners = {}
            for hotkey, revision, env in full_scan_combos:
                key = f"{hotkey}#{revision}"
                if key not in unique_miners:
                    unique_miners[key] = {'hotkey': hotkey, 'revision': revision}
            
            # Extract unique envs from full_scan_combos
            unique_envs = {env for _, _, env in full_scan_combos if env in current_env_ranges}
            
            full_scan_data = await sample_dao.get_scoring_samples_batch(
                miners=list(unique_miners.values()),
                env_ranges={e: current_env_ranges[e] for e in unique_envs}
            )
            new_samples.update(full_scan_data)
            
            # Collect timestamps from full scan
            for miner_samples in full_scan_data.values():
                for env_samples in miner_samples.values():
                    new_timestamps.extend(s['timestamp'] for s in env_samples)
        
        if incr_scan_combos:
            # All incremental scans use the same global timestamp
            logger.info(f"Incremental scan: {len(incr_scan_combos)} combos since {_format_timestamp(self._last_sample_ts)}")
            
            incr_data = await sample_dao.get_scoring_samples_incremental(
                changed_combos=incr_scan_combos,
                env_ranges=current_env_ranges,
                since_timestamp=self._last_sample_ts
            )
            
            # Log query results
            total_incr_samples = sum(len(samples) for env_data in incr_data.values() for samples in env_data.values())
            logger.debug(f"Retrieved {total_incr_samples} samples from {len(incr_data)} miners")
            
            # Merge incremental results
            for key, envs_data in incr_data.items():
                if key not in new_samples:
                    new_samples[key] = {}
                new_samples[key].update(envs_data)
            
            # Collect timestamps from incremental scan
            for miner_samples in incr_data.values():
                for env_samples in miner_samples.values():
                    new_timestamps.extend(s['timestamp'] for s in env_samples)
        
        # Step 4: Calculate update statistics before merge
        total_new_samples = 0
        miner_update_details = []
        
        for key, envs_data in new_samples.items():
            miner_sample_count = sum(len(samples) for samples in envs_data.values())
            total_new_samples += miner_sample_count
            if miner_sample_count > 0:
                miner_update_details.append(f"{key}: {miner_sample_count} samples")
        
        # Step 5: Merge with cache
        if new_samples:
            updated_data = self._merge_incremental(self._data, new_samples, current_env_ranges, valid_miners, current_envs)
            self._data = updated_data
        
        # Step 6: Update metadata
        self._cached_envs = current_envs
        self._cached_env_ranges = current_env_ranges
        self._cached_miners = current_miners
        
        # Update global max timestamp from this refresh
        if new_timestamps:
            self._last_sample_ts = max(new_timestamps)
        
        # Log detailed update statistics
        logger.info(f"Fine-grained refresh completed: {len(new_samples)} miners updated, {total_new_samples} total samples")
        if miner_update_details:
            logger.debug(f"Update details: {', '.join(miner_update_details[:10])}" + (f" ... ({len(miner_update_details) - 10} more)" if len(miner_update_details) > 10 else ""))
    
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
    
    def _merge_incremental(
        self,
        old_data: dict,
        new_samples: dict,
        env_ranges: dict,
        valid_miners: list,
        current_envs: list
    ) -> Dict[str, Any]:
        """Merge incremental samples into old cache.
        
        Rebuilds cache structure with current miner list to handle:
        - New miners added
        - Removed miners deleted
        - New environments added
        - Updated samples for existing miners
        """
        # Rebuild cache structure from current valid miners (automatically removes old miners)
        updated_data = {}
        
        for miner in valid_miners:
            uid = str(miner['uid'])
            # Restore from old cache if exists, otherwise initialize
            if uid in old_data:
                updated_data[uid] = old_data[uid]
            else:
                # New miner - initialize structure
                updated_data[uid] = {
                    'hotkey': miner['hotkey'],
                    'model_revision': miner['revision'],
                    'model_repo': miner.get('model'),
                    'first_block': miner.get('first_block'),
                    'env': {}
                }
        
        # Ensure all environments exist in each miner's structure
        for miner_entry in updated_data.values():
            for env in current_envs:
                if env not in miner_entry['env']:
                    # New env - initialize with empty samples
                    start_id, end_id = env_ranges.get(env, (0, 0))
                    miner_entry['env'][env] = {
                        'samples': [],
                        'total_count': end_id - start_id if start_id < end_id else 0,
                        'completed_count': 0,
                        'missing_task_ids': list(range(start_id, end_id))[:100] if start_id < end_id else [],
                        'completeness': 0.0
                    }
        
        # Merge new samples into structure
        for miner_entry in updated_data.values():
            hotkey = miner_entry['hotkey']
            revision = miner_entry['model_revision']
            key = f"{hotkey}#{revision}"
            
            if key not in new_samples:
                continue
            
            for env, new_env_samples in new_samples[key].items():
                if env not in miner_entry['env']:
                    continue
                
                # Merge samples
                old_samples_dict = {s['task_id']: s for s in miner_entry['env'][env]['samples']}
                
                for new_sample in new_env_samples:
                    task_id = new_sample['task_id']
                    # Keep latest timestamp
                    if task_id not in old_samples_dict or new_sample['timestamp'] > old_samples_dict[task_id]['timestamp']:
                        old_samples_dict[task_id] = {
                            'task_id': task_id,
                            'score': new_sample['score'],
                            'task_uuid': new_sample['timestamp'],
                            'timestamp': new_sample['timestamp'],
                        }
                
                merged_samples = list(old_samples_dict.values())
                
                # Filter out samples outside the new range (handles range shrinkage)
                start_id, end_id = env_ranges.get(env, (0, 0))
                merged_samples = [s for s in merged_samples if start_id <= s['task_id'] < end_id]
                
                # Recalculate stats
                expected_count = end_id - start_id
                completed_count = len(merged_samples)
                completeness = completed_count / expected_count if expected_count > 0 else 0.0
                
                completed_task_ids = {s['task_id'] for s in merged_samples}
                all_task_ids = set(range(start_id, end_id))
                missing_task_ids = sorted(list(all_task_ids - completed_task_ids))[:100]
                
                miner_entry['env'][env] = {
                    'samples': merged_samples,
                    'total_count': expected_count,
                    'completed_count': completed_count,
                    'missing_task_ids': missing_task_ids,
                    'completeness': round(completeness, 4)
                }
        
        return updated_data
    
    def _remove_envs_from_cache(self, removed_envs: set) -> None:
        """Remove deleted environments from cache data."""
        logger.info(f"Removing {len(removed_envs)} envs from cache: {removed_envs}")
        
        for miner_entry in self._data.values():
            for env in removed_envs:
                if env in miner_entry['env']:
                    del miner_entry['env'][env]
    


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