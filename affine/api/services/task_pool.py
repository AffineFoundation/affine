"""
Task Pool Manager

Implements weighted random task selection and in-memory lock management.

Key Features:
- Weighted random selection: probability proportional to pending task count per miner
- In-memory locks: prevent multiple executors from getting same task
- Lock timeout: auto-release expired locks
- Cleanup: remove tasks for invalid miners
- Idempotent completion: gracefully handle already-completed/deleted tasks
"""

import asyncio
import time
import random
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

from affine.database.dao.task_queue import TaskQueueDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.miners import MinersDAO

from affine.core.setup import logger


class TaskLock:
    """Represents a task lock for preventing concurrent fetch of same task."""
    
    def __init__(self, task_uuid: str, executor_hotkey: str, timeout_seconds: int = 300):
        self.task_uuid = task_uuid
        self.executor_hotkey = executor_hotkey
        self.locked_at = time.time()
        self.timeout_seconds = timeout_seconds
    
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        return time.time() - self.locked_at > self.timeout_seconds
    
    def owned_by(self, hotkey: str) -> bool:
        """Check if lock is owned by given hotkey."""
        return self.executor_hotkey == hotkey


class AssignedTask:
    """Tracks assigned tasks with timeout for auto-reset."""
    
    def __init__(self, task_uuid: str, assigned_to: str, assigned_at: int, timeout_seconds: int = 600):
        self.task_uuid = task_uuid
        self.assigned_to = assigned_to
        self.assigned_at = assigned_at
        self.timeout_seconds = timeout_seconds
    
    def is_expired(self) -> bool:
        """Check if assignment has expired (10 minutes)."""
        return time.time() - self.assigned_at > self.timeout_seconds


class TaskPoolManager:
    """
    Manages task pool with weighted random selection and distributed locking.
    
    This is a singleton service that should be initialized once at API server startup.
    """
    
    _instance: Optional['TaskPoolManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(
        self,
        lock_timeout_seconds: int = 300,  # 5 minutes
        assigned_timeout_seconds: int = 600,  # 10 minutes
        cleanup_interval_seconds: int = 60,  # 1 minute
    ):
        """
        Initialize TaskPoolManager.
        
        Args:
            lock_timeout_seconds: Lock timeout in seconds (default: 300 = 5 min)
            assigned_timeout_seconds: Assigned task timeout (default: 600 = 10 min)
            cleanup_interval_seconds: Interval for cleanup tasks (default: 60s)
        """
        self.dao = TaskQueueDAO()
        self.logs_dao = ExecutionLogsDAO()
        self.miners_dao = MinersDAO()
        
        # In-memory lock storage for fetch concurrency control: {task_uuid: TaskLock}
        self.locks: Dict[str, TaskLock] = {}
        self.locks_lock = asyncio.Lock()
        
        # In-memory tracking of assigned tasks for timeout detection: {task_uuid: AssignedTask}
        self.assigned_tasks: Dict[str, AssignedTask] = {}
        self.assigned_tasks_lock = asyncio.Lock()
        
        self.lock_timeout_seconds = lock_timeout_seconds
        self.assigned_timeout_seconds = assigned_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Background task handles
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"TaskPoolManager initialized (lock_timeout={lock_timeout_seconds}s, "
            f"assigned_timeout={assigned_timeout_seconds}s, cleanup_interval={cleanup_interval_seconds}s)"
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
        """Start background cleanup tasks and load assigned tasks from database."""
        if self._running:
            logger.warning("Background tasks already running")
            return
        
        # Load assigned tasks from database on startup
        await self._load_assigned_tasks_from_db()
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TaskPoolManager background tasks started")
    
    async def _load_assigned_tasks_from_db(self):
        """Load all assigned tasks from database on startup/restart."""
        try:
            from affine.database.client import get_client
            client = get_client()
            
            # Scan for all assigned tasks across all environments
            params = {
                'TableName': self.dao.table_name,
                'FilterExpression': '#status = :assigned',
                'ExpressionAttributeNames': {'#status': 'status'},
                'ExpressionAttributeValues': {':assigned': {'S': 'assigned'}}
            }
            
            response = await client.scan(**params)
            tasks = [self.dao._deserialize(item) for item in response.get('Items', [])]
            
            async with self.assigned_tasks_lock:
                for task in tasks:
                    self.assigned_tasks[task['task_uuid']] = AssignedTask(
                        task_uuid=task['task_uuid'],
                        assigned_to=task.get('assigned_to', ''),
                        assigned_at=task.get('assigned_at', int(time.time())),
                        timeout_seconds=self.assigned_timeout_seconds
                    )
            
            logger.info(f"Loaded {len(tasks)} assigned tasks from database on startup")
            
        except Exception as e:
            logger.error(f"Error loading assigned tasks from database: {e}", exc_info=True)
    
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
        """Background loop for periodic cleanup."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                
                # Release expired fetch locks
                released_count = await self.release_expired_locks()
                if released_count > 0:
                    logger.info(f"Released {released_count} expired fetch locks")
                
                # Reset expired assigned tasks to pending
                reset_count = await self.reset_expired_assigned_tasks()
                if reset_count > 0:
                    logger.info(f"Reset {reset_count} expired assigned tasks to pending")
                
                # Cleanup invalid tasks (every 5 minutes)
                if int(time.time()) % 300 < self.cleanup_interval_seconds:
                    cleaned_count = await self.cleanup_invalid_tasks()
                    if cleaned_count > 0:
                        logger.info(f"Cleaned up {cleaned_count} invalid tasks")
                        
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
    
    async def reset_expired_assigned_tasks(self) -> int:
        """
        Reset expired assigned tasks to pending status.
        
        This prevents tasks from being permanently stuck in assigned state
        if executor crashes or takes too long (> 10 minutes).
        
        Returns:
            Number of tasks reset
        """
        reset_count = 0
        
        async with self.assigned_tasks_lock:
            expired_uuids = [
                uuid for uuid, assigned in self.assigned_tasks.items()
                if assigned.is_expired()
            ]
        
        for task_uuid in expired_uuids:
            try:
                # Get task from database
                task = await self.dao.get_task_by_uuid(task_uuid)
                
                if not task:
                    # Task no longer exists (already completed), remove from tracking
                    async with self.assigned_tasks_lock:
                        del self.assigned_tasks[task_uuid]
                    continue
                
                if task.get('status') != 'assigned':
                    # Task status changed, remove from tracking
                    async with self.assigned_tasks_lock:
                        del self.assigned_tasks[task_uuid]
                    continue
                
                # Reset task to pending status
                await self.dao.delete(task['pk'], task['sk'])
                
                task['sk'] = self.dao._make_sk('pending', task['task_uuid'])
                task['status'] = 'pending'
                task['assigned_to'] = None
                task['assigned_at'] = None
                
                await self.dao.put(task)
                
                # Remove from assigned tasks tracking
                async with self.assigned_tasks_lock:
                    del self.assigned_tasks[task_uuid]
                
                reset_count += 1
                logger.warning(
                    f"Reset expired assigned task {task_uuid} to pending "
                    f"(was assigned to {task.get('assigned_to')}, "
                    f"env={task.get('env')}, task_id={task.get('task_id')})"
                )
                
            except Exception as e:
                logger.error(f"Error resetting expired task {task_uuid}: {e}", exc_info=True)
        
        return reset_count
    
    async def cleanup_invalid_tasks(self) -> int:
        """
        Remove tasks for miners that are no longer valid.
        
        Directly queries MinersDAO to determine validity.
        
        Returns:
            Number of tasks cleaned up
        """
        try:
            # Get valid miners from DAO directly
            valid_miners_data = await self.miners_dao.get_valid_miners()
            
            if not valid_miners_data:
                logger.warning("No valid miners found, skipping cleanup")
                return 0

            # Convert to format expected by TaskQueueDAO
            valid_miners_list = [
                {
                    'hotkey': miner['hotkey'],
                    'model_revision': miner['revision']
                }
                for miner in valid_miners_data
            ]

            # Call DAO cleanup method
            cleaned_count = await self.dao.cleanup_invalid_tasks(valid_miners_list)
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up invalid tasks: {e}", exc_info=True)
            return 0
    
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
            
            logger.debug(f"Selected miner {selected_miner} (weights: {dict(zip(miners, weights))})")
            
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
            
            # Acquire lock
            async with self.locks_lock:
                # Double-check not locked (race condition prevention)
                if selected_task['task_uuid'] in self.locks:
                    logger.warning(f"Task {selected_task['task_uuid']} already locked, retrying")
                    return await self.fetch_task(executor_hotkey, env)  # Retry
                
                # Create lock
                lock = TaskLock(
                    selected_task['task_uuid'],
                    executor_hotkey,
                    self.lock_timeout_seconds
                )
                self.locks[selected_task['task_uuid']] = lock
            
            # Persist assignment to database
            assigned_task = await self.dao.assign_task(selected_task, executor_hotkey)
            
            # Track assigned task for timeout detection
            async with self.assigned_tasks_lock:
                self.assigned_tasks[assigned_task['task_uuid']] = AssignedTask(
                    task_uuid=assigned_task['task_uuid'],
                    assigned_to=executor_hotkey,
                    assigned_at=assigned_task['assigned_at'],
                    timeout_seconds=self.assigned_timeout_seconds
                )
            
            logger.info(
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
            # Step 1: Verify lock ownership
            async with self.locks_lock:
                lock = self.locks.get(task_uuid)
                
                if not lock:
                    logger.warning(f"Task {task_uuid} not locked, may be expired or already completed")
                    # Don't fail - could be duplicate completion or expired lock
                    # Fall through to check task existence
                elif not lock.owned_by(executor_hotkey):
                    logger.error(
                        f"Task {task_uuid} lock mismatch: "
                        f"owned by {lock.executor_hotkey}, completed by {executor_hotkey}"
                    )
                    return {
                        'status': 'error',
                        'message': f'Lock owned by different executor'
                    }
                
                # Release lock
                if lock:
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
                # Delete task from queue
                await self.dao.complete_task(task)
                
                # Remove from assigned tasks tracking
                async with self.assigned_tasks_lock:
                    if task_uuid in self.assigned_tasks:
                        del self.assigned_tasks[task_uuid]
                
                logger.info(
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
                
                # Update assigned tasks tracking
                async with self.assigned_tasks_lock:
                    if updated_task['status'] == 'pending':
                        # Reset to pending for retry, remove from tracking (will be reassigned)
                        if task_uuid in self.assigned_tasks:
                            del self.assigned_tasks[task_uuid]
                    elif updated_task['status'] == 'failed':
                        # Permanently failed, remove from tracking
                        if task_uuid in self.assigned_tasks:
                            del self.assigned_tasks[task_uuid]
                
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