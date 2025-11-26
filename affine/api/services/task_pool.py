"""
Task Pool Manager

Implements weighted random task selection with minimal locking.

Key Features:
- Weighted random selection: probability proportional to pending task count per miner
- Lightweight in-memory locks: prevent concurrent fetch within same process
- Idempotent completion: gracefully handle already-completed/deleted tasks
"""

import asyncio
import time
import random
from typing import Dict, Any, Optional, List
from collections import defaultdict

from affine.database.dao.task_pool import TaskPoolDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO

from affine.core.setup import logger


class TaskLock:
    """Lightweight lock for preventing concurrent fetch within same process."""
    
    def __init__(self, task_uuid: str, timeout_seconds: int = 30):
        self.task_uuid = task_uuid
        self.locked_at = time.time()
        self.timeout_seconds = timeout_seconds
    
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        return time.time() - self.locked_at > self.timeout_seconds


class TaskPoolManager:
    """
    Manages task pool with weighted random selection.
    
    Singleton service initialized once at API server startup.
    """
    
    _instance: Optional['TaskPoolManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(
        self,
        lock_timeout_seconds: int = 30,  # 30 seconds
        cleanup_interval_seconds: int = 60,  # 1 minute
    ):
        """
        Initialize TaskPoolManager.
        
        Args:
            lock_timeout_seconds: Lock timeout in seconds (default: 30s)
            cleanup_interval_seconds: Interval for cleanup tasks (default: 60s)
        """
        self.dao = TaskPoolDAO()
        self.logs_dao = ExecutionLogsDAO()
        
        # Lightweight in-memory locks for fetch concurrency control
        self.locks: Dict[str, TaskLock] = {}
        self.locks_lock = asyncio.Lock()
        
        self.lock_timeout_seconds = lock_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Background task handles
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"TaskPoolManager initialized (lock_timeout={lock_timeout_seconds}s, "
            f"cleanup_interval={cleanup_interval_seconds}s)"
        )
    
    @classmethod
    def get_instance(cls) -> 'TaskPoolManager':
        """Get singleton instance (lazy initialization)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    async def initialize(cls, **kwargs) -> 'TaskPoolManager':
        """
        Initialize singleton instance with custom parameters.
        
        Thread-safe initialization for API server startup.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
                await cls._instance.start_background_tasks()
        return cls._instance
    
    async def start_background_tasks(self):
        """Start background cleanup tasks."""
        if self._running:
            logger.warning("Background tasks already running")
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TaskPoolManager background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background cleanup tasks."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskPoolManager background tasks stopped")
    
    async def _cleanup_loop(self):
        """Background loop for periodic cleanup of expired locks."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                
                # Release expired fetch locks
                released_count = await self.release_expired_locks()
                if released_count > 0:
                    logger.debug(f"Released {released_count} expired fetch locks")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
    
    async def release_expired_locks(self) -> int:
        """
        Release all expired fetch locks (concurrency control).
        
        Returns:
            Number of locks released
        """
        async with self.locks_lock:
            expired_uuids = [
                uuid for uuid, lock in self.locks.items()
                if lock.is_expired()
            ]
            
            for uuid in expired_uuids:
                del self.locks[uuid]
                logger.debug(f"Released expired fetch lock for task {uuid}")
            
            return len(expired_uuids)
    
    async def fetch_task(
        self,
        executor_hotkey: str,
        env: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a task using weighted random selection.
        
        Selection algorithm:
        1. Get all pending tasks (excluding locked ones)
        2. Group by (hotkey, revision), count per miner
        3. Select miner with probability proportional to task count
        4. Randomly select one task from chosen miner
        5. Acquire lock for selected task
        
        Args:
            executor_hotkey: Executor's hotkey (for lock ownership)
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
            
            # Aggregate miner counts across all environments
            all_miner_counts: Dict[str, int] = defaultdict(int)
            env_tasks: Dict[str, List[Dict[str, Any]]] = {}
            
            for search_env in envs_to_search:
                pending_tasks = await self.dao.get_pending_tasks_by_env(search_env, limit=None)
                
                # Filter out locked tasks
                async with self.locks_lock:
                    available_tasks = [
                        task for task in pending_tasks
                        if task['task_uuid'] not in self.locks
                    ]
                
                if available_tasks:
                    env_tasks[search_env] = available_tasks
                    
                    # Count by miner
                    for task in available_tasks:
                        key = f"{task['miner_hotkey']}#{task['model_revision']}"
                        all_miner_counts[key] += 1
            
            if not all_miner_counts:
                logger.debug("No available tasks found")
                return None
            
            # Weighted random selection of miner
            miners = list(all_miner_counts.keys())
            weights = [all_miner_counts[m] for m in miners]
            selected_miner = random.choices(miners, weights=weights, k=1)[0]
            
            # logger.debug(f"Selected miner {selected_miner} (weights: {dict(zip(miners, weights))})")
            
            # Extract hotkey and revision
            hotkey, revision = selected_miner.split('#', 1)
            
            # Collect all tasks for selected miner across environments
            miner_tasks = []
            for search_env, tasks in env_tasks.items():
                miner_tasks.extend([
                    task for task in tasks
                    if task['miner_hotkey'] == hotkey and task['model_revision'] == revision
                ])
            
            if not miner_tasks:
                logger.warning(f"No tasks found for selected miner {selected_miner}")
                return None
            
            # Randomly select one task from miner's tasks
            selected_task = random.choice(miner_tasks)
            
            # Acquire lightweight lock (prevents concurrent fetch in same process)
            async with self.locks_lock:
                # Double-check not locked (race condition prevention)
                if selected_task['task_uuid'] in self.locks:
                    logger.warning(f"Task {selected_task['task_uuid']} already locked, retrying")
                    return await self.fetch_task(executor_hotkey, env)  # Retry
                
                # Create lock
                self.locks[selected_task['task_uuid']] = TaskLock(
                    selected_task['task_uuid'],
                    self.lock_timeout_seconds
                )
            
            # Persist assignment to database
            assigned_task = await self.dao.assign_task(selected_task, executor_hotkey)
            
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
            executor_hotkey: Executor's hotkey (for lock verification)
            success: Whether task succeeded
            result: Task result (for success case)
            error_message: Error message (for failure case)
            error_code: Error code (for failure case)
            
        Returns:
            Status dict with 'status' and 'message' keys
        """
        try:
            # Step 1: Release lock if exists
            async with self.locks_lock:
                if task_uuid in self.locks:
                    del self.locks[task_uuid]
            
            # Step 2: Check if task still exists
            task = await self.dao.get_task_by_uuid(task_uuid)
            
            if not task:
                # Task already deleted/completed - idempotent handling
                logger.info(
                    f"Task {task_uuid} already removed (completed/deleted), "
                    f"ignoring completion from {executor_hotkey}"
                )
                return {
                    'status': 'not_found',
                    'message': 'Task already completed or removed'
                }
            
            # Step 3: Log completion attempt
            if success:
                await self.logs_dao.log_task_complete(
                    miner_hotkey=task['miner_hotkey'],
                    task_uuid=task_uuid,
                    dataset_task_id=task['task_id'],
                    env=task['env'],
                    executor_hotkey=executor_hotkey,
                    score=result.get('score', 0.0) if result else 0.0,
                    latency_ms=result.get('latency_ms', 0) if result else 0,
                    execution_time_ms=result.get('execution_time_ms', 0) if result else 0
                )
            else:
                await self.logs_dao.log_task_failure(
                    miner_hotkey=task['miner_hotkey'],
                    task_uuid=task_uuid,
                    dataset_task_id=task['task_id'],
                    env=task['env'],
                    executor_hotkey=executor_hotkey,
                    error_message=error_message or 'Unknown error',
                    error_code=error_code or 'EXECUTION_ERROR',
                    error_type='execution',
                    execution_time_ms=0
                )
            
            # Step 4: Complete or fail task
            if success:
                # Delete task from pool
                await self.dao.complete_task(task)
                
                logger.debug(
                    f"Task {task_uuid} completed successfully by {executor_hotkey} "
                    f"(miner={task['miner_hotkey']}, env={task['env']}, task_id={task['task_id']})"
                )
                
                return {
                    'status': 'completed',
                    'message': 'Task completed successfully'
                }
            else:
                # Record failure and retry logic
                updated_task = await self.dao.fail_task(
                    task,
                    error_message or 'Unknown error',
                    error_code or 'EXECUTION_ERROR'
                )
                
                if updated_task['status'] == 'failed':
                    logger.warning(
                        f"Task {task_uuid} permanently failed after "
                        f"{updated_task['retry_count']} retries"
                    )
                else:
                    logger.info(
                        f"Task {task_uuid} failed, retry {updated_task['retry_count']} "
                        f"(max {updated_task['max_retries']})"
                    )
                
                return {
                    'status': 'failed',
                    'message': f"Task failed (retry {updated_task['retry_count']})"
                }
                
        except Exception as e:
            logger.error(f"Error completing task {task_uuid}: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Internal error: {str(e)}'
            }