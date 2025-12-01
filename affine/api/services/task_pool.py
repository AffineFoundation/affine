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
    """
    
    def __init__(self, count_cache_ttl: int = 30):
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
        
        logger.info(f"TaskPoolManager initialized (count_cache_ttl={count_cache_ttl}s)")
    
    async def _get_miner_counts(self, env: str) -> Dict[str, int]:
        """Get task counts per miner with caching.
        
        Cache TTL: 30s (configurable)
        Cold start: First call will query DB
        """
        async with self._cache_lock:
            # Check cache
            last_refresh = self._count_cache_ts.get(env, 0)
            age = time.time() - last_refresh

            if age > self._count_cache_ttl or env not in self._count_cache:
                # Refresh from DB
                logger.debug(f"Refreshing count cache for {env} (age={age:.1f}s)")
                counts = await self.dao.get_miner_task_counts(env)
                self._count_cache[env] = counts
                self._count_cache_ts[env] = time.time()
                return counts.copy()

            # Return cached
            return self._count_cache[env].copy()
    
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
        env: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a task using weighted random selection.
        
        Algorithm (two-phase query):
        1. Get task count per miner (lightweight COUNT query)
        2. Select miner weighted by task count
        3. Query selected miner's tasks (limit=10 per env)
        4. Randomly select one task
        5. Assign task atomically
        
        Args:
            executor_hotkey: Executor's hotkey
            env: Optional environment filter (if None, select from all envs)
            
        Returns:
            Task dict if available, None if no tasks available
        """
        try:
            # Determine environments to search
            if env:
                envs_to_search = [env]
            else:
                # Get all sampling environments from config
                from affine.api.dependencies import get_system_config
                config = await get_system_config()
                envs_to_search = config.get('sampling_environments', ['affine:sat'])
            
            # Phase 1: Get miner counts (cached)
            all_miner_counts: Dict[str, int] = {}
            
            for search_env in envs_to_search:
                counts = await self._get_miner_counts(search_env)
                
                # Aggregate across environments
                for miner_key, count in counts.items():
                    all_miner_counts[miner_key] = all_miner_counts.get(miner_key, 0) + count
            
            if not all_miner_counts:
                logger.debug("No available tasks found")
                return None
            
            # Phase 2: Weighted random selection of miner
            miners = list(all_miner_counts.keys())
            weights = [all_miner_counts[m] for m in miners]
            selected_miner = random.choices(miners, weights=weights, k=1)[0]
            
            # Extract hotkey and revision
            hotkey, revision = selected_miner.split('#', 1)
            
            # Phase 3: Get tasks for selected miner only (limit=10 per env)
            # We only need a few tasks since we'll randomly select one
            miner_tasks = []
            for search_env in envs_to_search:
                tasks = await self.dao.get_pending_tasks_for_miner(
                    search_env,
                    hotkey,
                    revision,
                    limit=10  # Fetch 10 tasks per env (enough for random selection)
                )
                miner_tasks.extend(tasks)
            
            if not miner_tasks:
                logger.warning(f"No tasks found for selected miner {selected_miner}")
                return None
            
            # Randomly select one task from miner's tasks
            selected_task = random.choice(miner_tasks)
            
            # Persist assignment to database (DynamoDB provides atomicity)
            assigned_task = await self.dao.assign_task(selected_task, executor_hotkey)
            
            # Cache UUID location for fast completion lookup
            async with self._cache_lock:
                self._uuid_cache[assigned_task['task_uuid']] = (
                    assigned_task['pk'],
                    assigned_task['sk']
                )
            
            logger.debug(
                f"Task {assigned_task['task_uuid']} assigned to {executor_hotkey} "
                f"(miner={hotkey}, env={assigned_task['env']}, task_id={assigned_task['task_id']})"
            )
            
            return assigned_task
            
        except Exception as e:
            logger.error(f"Error fetching task: {e}", exc_info=True)
            return None
    
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