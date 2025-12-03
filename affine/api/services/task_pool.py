"""
Task Pool Manager

Implements weighted random task selection with UUID caching.

Key Features:
- Weighted random selection: probability proportional to pending task count per miner
- UUID location cache: fast O(1) task lookup during completion
- Idempotent completion: gracefully handle already-completed/deleted tasks

Optimizations:
- No locking: DynamoDB provides atomicity via delete+put
- UUID cache: avoid expensive Scan operations (50x speedup)
- Dependency injection: consistent with other DAOs
"""

import asyncio
import time
import random
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar, Generic

from affine.database.dao.task_pool import TaskPoolDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.miners import MinersDAO

from affine.core.setup import logger


T = TypeVar('T')


class AsyncCache(Generic[T]):
    """Generic async cache with background refresh support.
    
    Features:
    - TTL-based expiration
    - Non-blocking background refresh
    - Cold start handling (blocks only on first fetch)
    """
    
    def __init__(self, ttl: int, name: str = "cache"):
        """Initialize cache.
        
        Args:
            ttl: Time-to-live in seconds
            name: Cache name for logging
        """
        self.ttl = ttl
        self.name = name
        self._data: Optional[T] = None
        self._timestamp: float = 0
        self._refresh_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def get(self, fetcher: Callable[[], T]) -> T:
        """Get cached data with background refresh.
        
        Args:
            fetcher: Async function to fetch fresh data
            
        Returns:
            Cached or fresh data
        """
        # Fast path: return cached data if available
        async with self._lock:
            if self._data is not None:
                age = time.time() - self._timestamp
                
                # Trigger background refresh if expired
                if age > self.ttl:
                    if self._refresh_task is None or self._refresh_task.done():
                        logger.debug(f"{self.name} cache expired (age={age:.1f}s), triggering refresh")
                        self._refresh_task = asyncio.create_task(
                            self._background_refresh(fetcher)
                        )
                
                # Return cached data (even if stale)
                return self._data
        
        # Cold start: block on first fetch
        logger.info(f"{self.name} cache cold start, fetching data")
        data = await fetcher()
        
        async with self._lock:
            self._data = data
            self._timestamp = time.time()
        
        return data
    
    async def _background_refresh(self, fetcher: Callable[[], T]):
        """Background task to refresh cache."""
        try:
            logger.debug(f"{self.name} cache background refresh started")
            start_time = time.time()
            
            data = await fetcher()
            
            elapsed = time.time() - start_time
            logger.debug(f"{self.name} cache refreshed in {elapsed:.2f}s")
            
            async with self._lock:
                self._data = data
                self._timestamp = time.time()
                
        except Exception as e:
            logger.error(f"{self.name} cache refresh failed: {e}", exc_info=True)


class TaskPoolManager:
    """
    Manages task pool with weighted random selection and dual caching.
    
    Uses background refresh for miner counts to avoid blocking fetch requests.
    """
    
    def __init__(self, count_cache_ttl: int = 30, miners_cache_ttl: int = 60):
        """Initialize TaskPoolManager with caches."""
        self.dao = TaskPoolDAO()
        self.logs_dao = ExecutionLogsDAO()
        self.miners_dao = MinersDAO()
        
        # Async caches with background refresh
        self._count_caches: Dict[str, AsyncCache[Dict[str, int]]] = {}  # {env: cache}
        self._count_cache_ttl = count_cache_ttl
        self._miners_cache = AsyncCache[Dict[str, Dict[str, Any]]](
            ttl=miners_cache_ttl,
            name="miners"
        )
        
        # UUID location cache: task_uuid -> (pk, sk)
        self._uuid_cache: Dict[str, Tuple[str, str]] = {}
        self._cache_lock = asyncio.Lock()
        
        logger.info(f"TaskPoolManager initialized (count_cache_ttl={count_cache_ttl}s, miners_cache_ttl={miners_cache_ttl}s)")
    
    async def _get_miner_counts(self, env: str) -> Dict[str, int]:
        """Get task counts per miner with non-blocking cache refresh."""
        # Create cache for this env if not exists
        if env not in self._count_caches:
            self._count_caches[env] = AsyncCache[Dict[str, int]](
                ttl=self._count_cache_ttl,
                name=f"count[{env}]"
            )
        
        return await self._count_caches[env].get(
            lambda: self.dao.get_miner_task_counts(env)
        )
    
    async def _get_miners(self) -> Dict[str, Dict[str, Any]]:
        """Get all miners with non-blocking cache refresh."""
        async def fetch_miners():
            miners_list = await self.miners_dao.get_all_miners()
            return {miner['hotkey']: miner for miner in miners_list}
        
        return await self._miners_cache.get(fetch_miners)
    
    def _select_miners_weighted(
        self,
        miner_counts: Dict[str, int],
        count: int,
        max_weight_ratio: float = 2.0
    ) -> List[str]:
        """Select miners using weighted random sampling without replacement.
        
        Anti-starvation: max probability ratio is capped at max_weight_ratio.
        
        Args:
            miner_counts: Dict mapping miner_key -> pending task count
            count: Number of miners to select
            max_weight_ratio: Max ratio between highest and lowest weight (default: 2.0)
        
        Returns:
            List of selected miner_keys (no duplicates, length <= count)
        """
        if not miner_counts:
            return []
        
        # Calculate base weight (handles min=0 case)
        counts = list(miner_counts.values())
        non_zero = [c for c in counts if c > 0]
        base_weight = min(non_zero) if non_zero else 1
        
        # Cap all weights: weight = min(task_count, base_weight * max_ratio)
        # Add base_weight to ensure min weight is not 0
        weights = [
            (key, min(count, base_weight * max_weight_ratio) + base_weight)
            for key, count in miner_counts.items()
        ]
        
        # Weighted sampling without replacement
        selected = []
        remaining = weights.copy()
        
        for _ in range(min(count, len(remaining))):
            total = sum(w for _, w in remaining)
            rand = random.uniform(0, total)
            
            cumulative = 0
            for item in remaining:
                cumulative += item[1]
                if rand <= cumulative:
                    selected.append(item[0])
                    remaining.remove(item)
                    break
        
        return selected
    
    async def _get_task_location(
        self, 
        task_uuid: str
    ) -> Optional[Tuple[str, str]]:
        """
        Get (PK, SK) for task UUID, with cache and DB fallback.
        
        Cache strategy:
        1. Check cache first (fast path)
        2. If miss, scan DB (cold start / evicted entry)
        3. Update cache for future lookups
        
        Args:
            task_uuid: Task UUID
            
        Returns:
            (pk, sk) tuple if found, None otherwise
        """
        # Fast path: check cache
        async with self._cache_lock:
            location = self._uuid_cache.get(task_uuid)
        
        if location:
            return location
        
        # Slow path: DB scan (cache miss)
        logger.debug(f"UUID cache miss for {task_uuid}, scanning DB")
        task = await self.dao.get_task_by_uuid(task_uuid)
        
        if not task:
            return None
        
        # Cache location for future lookups
        pk_sk = (task['pk'], task['sk'])
        async with self._cache_lock:
            self._uuid_cache[task_uuid] = pk_sk
            logger.debug(f"Cached location for {task_uuid}: {pk_sk}")
        
        return pk_sk
    
    async def fetch_task(
        self,
        executor_hotkey: str,
        env: Optional[str] = None,
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Fetch task(s) using weighted random selection with parallel queries.
        
        Algorithm:
        1. Select N miners randomly (weighted), where N = batch_size * 2-3
        2. Query all selected miners in parallel (asyncio.gather)
        3. Collect valid tasks and assign in parallel
        
        Args:
            executor_hotkey: Executor's hotkey
            env: Optional environment filter (if None, select from all envs)
            batch_size: Number of tasks to fetch (default: 1)
            
        Returns:
            List of task dicts (may be empty, length 0 to batch_size)
        """
        try:
            # Validate env parameter is provided
            if not env:
                logger.error("env parameter is required for fetch_task")
                return []
            
            # Get miner counts (cached)
            all_miner_counts = await self._get_miner_counts(env)
            
            if not all_miner_counts:
                logger.debug(f"No available tasks found for env={env}")
                return []
            
            # Weighted random selection of miners (without replacement)
            num_miners_to_query = min(int(batch_size * 1.5), len(all_miner_counts))
            
            # Select miners using weighted random sampling (anti-starvation: max_ratio=2.0)
            selected_miners = self._select_miners_weighted(
                all_miner_counts,
                num_miners_to_query,
                max_weight_ratio=2.0
            )
            
            # Parallel query all selected miners
            query_tasks = []
            for miner_key in selected_miners:
                hotkey, revision = miner_key.split('#', 1)
                query_tasks.append(
                    self.dao.get_pending_tasks_for_miner(env, hotkey, revision, limit=1)
                )
            
            # Execute all queries in parallel
            results = await asyncio.gather(*query_tasks, return_exceptions=True)
            
            # Collect valid tasks (filter out exceptions and empty results)
            candidate_tasks = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                if isinstance(result, list) and result:
                    candidate_tasks.append(result[0])  # Take first task
            
            # Take first batch_size tasks and assign in parallel
            tasks_to_assign = candidate_tasks[:batch_size]
            
            if not tasks_to_assign:
                logger.debug(f"No valid tasks found for env={env} after querying {num_miners_to_query} miners")
                return []
            
            # Parallel assignment
            assign_tasks = [
                self.dao.assign_task(task, executor_hotkey)
                for task in tasks_to_assign
            ]
            assigned_results = await asyncio.gather(*assign_tasks, return_exceptions=True)
            
            # Filter successful assignments, cache UUIDs, and enrich with miner data
            miners_dict = await self._get_miners()
            
            assigned_tasks = []
            for result in assigned_results:
                if isinstance(result, Exception):
                    logger.warning(f"Failed to assign task: {result}")
                    continue
                
                # Cache UUID location
                async with self._cache_lock:
                    self._uuid_cache[result['task_uuid']] = (
                        result['pk'],
                        result['sk']
                    )
                
                # Enrich task with miner data from cache
                miner_hotkey = result['miner_hotkey']
                miner_record = miners_dict.get(miner_hotkey)
                
                if not miner_record:
                    logger.warning(f"Miner record not found for hotkey {miner_hotkey[:16]}..., skipping task")
                    continue
                
                miner_uid = miner_record.get('uid')
                if miner_uid is None:
                    logger.warning(f"UID not found for hotkey {miner_hotkey[:16]}..., skipping task")
                    continue
                
                chute_slug = miner_record.get('chute_slug')
                if not chute_slug:
                    logger.warning(f"chute_slug not found for hotkey {miner_hotkey[:16]}..., skipping task")
                    continue
                
                # Add miner_uid and chute_slug to task
                enriched_task = {
                    **result,
                    'miner_uid': miner_uid,
                    'chute_slug': chute_slug,
                }
                
                assigned_tasks.append(enriched_task)
                
                logger.debug(
                    f"Task {result['task_uuid']} assigned to {executor_hotkey} "
                    f"(miner={miner_hotkey[:12]}..., uid={miner_uid}, env={env}, task_id={result['task_id']})"
                )
            
            logger.info(
                f"TaskPoolManager.fetch_task({env}): "
                f"queried {num_miners_to_query} miners, assigned {len(assigned_tasks)}/{batch_size} tasks"
            )
            
            # Always return list
            return assigned_tasks
            
        except Exception as e:
            logger.error(f"Error fetching task(s): {e}", exc_info=True)
            return []
    
    async def complete_task(
        self,
        task_uuid: str,
        executor_hotkey: str,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Complete a task (success or failure).
        
        Idempotent: if task already completed/deleted, just log and return success.
        
        Args:
            task_uuid: Task UUID
            executor_hotkey: Executor's hotkey
            success: Whether task succeeded
            result: Task result (for success case)
            error_message: Error message (for failure case)
            error_code: Error code (for failure case)
            
        Returns:
            Status dict with 'status' and 'message' keys
        """
        try:
            # Step 1: Get task location (with cache)
            location = await self._get_task_location(task_uuid)
            
            if not location:
                logger.info(
                    f"Task {task_uuid} not found (completed/deleted), "
                    f"ignoring completion from {executor_hotkey}"
                )
                return {
                    'status': 'not_found',
                    'message': 'Task already completed or removed'
                }
            
            pk, sk = location
            
            # Step 2: Get full task data
            task = await self.dao.get(pk, sk)
            
            if not task:
                # Race condition: task deleted between cache check and get
                logger.warning(
                    f"Task {task_uuid} deleted after cache lookup "
                    f"(race condition, ignoring)"
                )
                
                # Clean cache
                async with self._cache_lock:
                    self._uuid_cache.pop(task_uuid, None)
                
                return {
                    'status': 'not_found',
                    'message': 'Task already completed or removed'
                }
            
            # Step 3: Log completion attempt
            if success:
                if not result:
                    raise ValueError(
                        f"Task {task_uuid} marked as success but result is None. "
                        "This indicates a bug in the caller."
                    )
                
                await self.logs_dao.log_task_complete(
                    miner_hotkey=task['miner_hotkey'],
                    task_uuid=task_uuid,
                    dataset_task_id=task['task_id'],
                    env=task['env'],
                    executor_hotkey=executor_hotkey,
                    score=result['score'],
                    latency_ms=result['latency_ms'],
                    execution_time_ms=result.get('execution_time_ms', 0)
                )
            else:
                if not error_message:
                    raise ValueError(
                        f"Task {task_uuid} marked as failure but error_message is None. "
                        "This indicates a bug in the caller."
                    )
                
                await self.logs_dao.log_task_failure(
                    miner_hotkey=task['miner_hotkey'],
                    task_uuid=task_uuid,
                    dataset_task_id=task['task_id'],
                    env=task['env'],
                    executor_hotkey=executor_hotkey,
                    error_message=error_message,
                    error_code=error_code,
                    error_type='execution',
                    execution_time_ms=0
                )
            
            # Step 4: Complete or fail task
            if success:
                # Delete task from pool
                await self.dao.complete_task(task)
                
                # Remove from cache
                async with self._cache_lock:
                    self._uuid_cache.pop(task_uuid, None)
                
                logger.debug(
                    f"Task {task_uuid} completed successfully by {executor_hotkey} "
                    f"(miner={task['miner_hotkey']}, env={task['env']}, task_id={task['task_id']})"
                )
                
                return {
                    'status': 'completed',
                    'message': 'Task completed successfully'
                }
            
            # Handle task failure
            # error_message already validated above
            updated_task = await self.dao.fail_task(
                task,
                error_message,
                error_code
            )

            # fail_task() returns either 'deleted' or 'pending' status
            if updated_task['status'] == 'deleted':
                # Max retries reached, permanently deleted
                async with self._cache_lock:
                    self._uuid_cache.pop(task_uuid, None)
                
                logger.warning(
                    f"Task {task_uuid} permanently deleted after "
                    f"{updated_task['retry_count']} retries (max={updated_task['max_retries']})"
                )
                return {
                    'status': 'deleted',
                    'message': f"Task permanently deleted after {updated_task['retry_count']} retries"
                }
            
            # Status is 'pending', will retry
            async with self._cache_lock:
                self._uuid_cache[task_uuid] = (updated_task['pk'], updated_task['sk'])
            
            logger.info(
                f"Task {task_uuid} will retry ({updated_task['retry_count']}/{updated_task['max_retries']})"
            )
            return {
                'status': 'retry',
                'message': f"Task will be retried ({updated_task['retry_count']}/{updated_task['max_retries']})"
            }
                
        except Exception as e:
            logger.error(f"Error completing task {task_uuid}: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}'
            }