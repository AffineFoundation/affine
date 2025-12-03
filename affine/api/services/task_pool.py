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
from typing import Dict, Any, Optional, List, Tuple

from affine.database.dao.task_pool import TaskPoolDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO

from affine.core.setup import logger


class TaskPoolManager:
    """
    Manages task pool with weighted random selection and dual caching.
    
    Uses background refresh for miner counts to avoid blocking fetch requests.
    """
    
    def __init__(self, count_cache_ttl: int = 20):
        """Initialize TaskPoolManager with caches."""
        self.dao = TaskPoolDAO()
        self.logs_dao = ExecutionLogsDAO()
        
        # Task count cache: {env: {miner_key: count}}
        self._count_cache: Dict[str, Dict[str, int]] = {}
        self._count_cache_ts: Dict[str, float] = {}
        self._count_cache_ttl = count_cache_ttl
        
        # UUID location cache: task_uuid -> (pk, sk)
        self._uuid_cache: Dict[str, Tuple[str, str]] = {}
        self._cache_lock = asyncio.Lock()
        
        # Background refresh control
        self._refresh_tasks: Dict[str, asyncio.Task] = {}  # {env: refresh_task}
        self._environments: set = set()  # Track known environments
        
        logger.info(f"TaskPoolManager initialized (count_cache_ttl={count_cache_ttl}s)")
    
    async def _get_miner_counts(self, env: str) -> Dict[str, int]:
        """Get task counts per miner with non-blocking cache refresh.
        
        Strategy:
        1. Return cached data immediately if available (even if stale)
        2. Trigger background refresh if cache is expired
        3. Only block on first call (cold start)
        
        This ensures fetch_task() never waits for slow COUNT queries.
        """
        # Track this environment for future background refreshes
        self._environments.add(env)
        
        # Fast path: return cached data if available
        async with self._cache_lock:
            if env in self._count_cache:
                last_refresh = self._count_cache_ts.get(env, 0)
                age = time.time() - last_refresh
                
                # Return stale cache immediately, trigger background refresh if needed
                if age > self._count_cache_ttl:
                    # Trigger background refresh without waiting
                    if env not in self._refresh_tasks or self._refresh_tasks[env].done():
                        logger.debug(f"Triggering background refresh for {env} (age={age:.1f}s)")
                        self._refresh_tasks[env] = asyncio.create_task(
                            self._background_refresh_counts(env)
                        )

                # Return cached data (even if stale)
                return self._count_cache[env].copy()
        
        # Cold start: no cache available, must wait for first query
        logger.info(f"Cold start: fetching miner counts for {env} (blocking)")
        counts = await self.dao.get_miner_task_counts(env)
        
        async with self._cache_lock:
            self._count_cache[env] = counts
            self._count_cache_ts[env] = time.time()
        
        return counts.copy()
    
    async def _background_refresh_counts(self, env: str):
        """Background task to refresh miner counts for an environment.
        
        This runs asynchronously without blocking fetch_task() calls.
        """
        try:
            logger.debug(f"Background refresh started for {env}")
            start_time = time.time()
            
            counts = await self.dao.get_miner_task_counts(env)
            
            elapsed = time.time() - start_time
            logger.debug(f"Background refresh completed for {env} in {elapsed:.2f}s")
            
            async with self._cache_lock:
                self._count_cache[env] = counts
                self._count_cache_ts[env] = time.time()
                
        except Exception as e:
            logger.error(f"Background refresh failed for {env}: {e}", exc_info=True)
    
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
            
            # Phase 1: Get miner counts (cached)
            all_miner_counts = await self._get_miner_counts(env)
            
            if not all_miner_counts:
                logger.debug(f"No available tasks found for env={env}")
                return []
            
            # Phase 2: Random selection of miners (without replacement)
            miners = list(all_miner_counts.keys())
            
            # Shuffle miners to randomize selection
            random.shuffle(miners)
            
            # Select first N miners (N = batch_size * 2 to handle miners with no tasks)
            # Cap at total available miners
            num_miners_to_query = min(batch_size * 2, len(miners))
            selected_miners = miners[:num_miners_to_query]
            
            # Phase 3: Parallel query all selected miners
            query_tasks = []
            miner_infos = []  # Track which miner each query corresponds to
            
            for miner_key in selected_miners:
                hotkey, revision = miner_key.split('#', 1)
                query_tasks.append(
                    self.dao.get_pending_tasks_for_miner(env, hotkey, revision, limit=1)
                )
                miner_infos.append((miner_key, hotkey, revision))
            
            # Execute all queries in parallel (100x speedup: 15s -> 150ms)
            results = await asyncio.gather(*query_tasks, return_exceptions=True)
            
            # Phase 4: Collect valid tasks (filter out exceptions and empty results)
            candidate_tasks = []
            for result, (miner_key, hotkey, revision) in zip(results, miner_infos):
                if isinstance(result, Exception):
                    logger.debug(f"Error querying miner {hotkey[:12]}...: {result}")
                    continue
                if isinstance(result, list) and result:
                    candidate_tasks.append(result[0])  # Take first task
            
            # Phase 5: Take first batch_size tasks and assign in parallel
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
            
            # Phase 6: Filter successful assignments and cache UUIDs
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
                
                assigned_tasks.append(result)
                
                logger.debug(
                    f"Task {result['task_uuid']} assigned to {executor_hotkey} "
                    f"(miner={result['miner_hotkey'][:12]}..., env={env}, task_id={result['task_id']})"
                )
            
            logger.info(
                f"Fetched {len(assigned_tasks)}/{batch_size} tasks for {env} "
                f"(queried {num_miners_to_query} miners in parallel)"
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