"""
Task Queue DAO

Manages sampling tasks with status tracking and error handling.

Design:
- PK: ENV#{env} - partition by environment for load distribution
- SK: STATUS#{status}#PRIORITY#{priority}#CREATED#{timestamp}#UUID#{uuid}
- task_id is an integer representing the dataset index (0 to dataset_length-1)
- Uses GSI for querying by miner+revision
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Set
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class TaskQueueDAO(BaseDAO):
    """DAO for task_queue table.
    
    Manages sampling tasks for miners with proper dataset index task_id.
    
    Schema Design:
    - PK: ENV#{env} - partition by environment
    - SK: STATUS#{status}#PRIORITY#{priority}#CREATED#{timestamp}#UUID#{uuid}
    - GSI1: MINER#{hotkey}#REV#{revision} -> ENV#{env}#TASK_ID#{task_id}
    
    task_id is an INTEGER representing the dataset index (0 to dataset_length-1),
    NOT a timestamp-uuid string.
    """
    
    def __init__(self):
        self.table_name = get_table_name("task_queue")
        super().__init__()
    
    def _make_pk(self, env: str) -> str:
        """Generate partition key based on environment."""
        return f"ENV#{env}"
    
    def _make_sk(self, status: str, priority: int, created_at: int, task_uuid: str) -> str:
        """Generate sort key with status, priority, timestamp, and uuid.
        
        Format allows:
        - Querying by status prefix
        - Sorting by priority (padded for lexicographic order)
        - Sorting by creation time within same priority
        - Unique identification via uuid
        """
        # Pad priority to 10 digits for proper sorting (higher priority = higher number)
        priority_padded = str(priority).zfill(10)
        return f"STATUS#{status}#PRIORITY#{priority_padded}#CREATED#{created_at}#UUID#{task_uuid}"
    
    def _make_gsi1_pk(self, miner_hotkey: str, model_revision: str) -> str:
        """Generate GSI1 partition key for miner+revision queries."""
        return f"MINER#{miner_hotkey}#REV#{model_revision}"
    
    def _make_gsi1_sk(self, env: str, task_id: int) -> str:
        """Generate GSI1 sort key for env+task_id queries."""
        return f"ENV#{env}#TASK_ID#{task_id}"
    
    async def create_task(
        self,
        miner_hotkey: str,
        model_revision: str,
        model: str,
        env: str,
        task_id: int,  # Dataset index, NOT uuid
        priority: int = 1000,
        ttl_days: int = 7
    ) -> Dict[str, Any]:
        """Create a new sampling task.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            model: Model repo/name
            env: Environment name (e.g., affine:sat, agentgym:webshop)
            task_id: Dataset index (0 to dataset_length-1)
            priority: Task priority (higher = more urgent, default 1000)
            ttl_days: Days until task expires
            
        Returns:
            Created task
        """
        task_uuid = str(uuid.uuid4())
        created_at = int(time.time())
        status = 'pending'
        
        item = {
            'pk': self._make_pk(env),
            'sk': self._make_sk(status, priority, created_at, task_uuid),
            'task_uuid': task_uuid,
            'task_id': task_id,  # Integer dataset index
            'miner_hotkey': miner_hotkey,
            'model_revision': model_revision,
            'model': model,
            'env': env,
            'status': status,
            'priority': priority,
            'created_at': created_at,
            'assigned_to': None,
            'assigned_at': None,
            'retry_count': 0,
            'max_retries': 3,
            'last_error': None,
            'last_error_code': None,
            'last_failed_at': None,
            'ttl': self.get_ttl(ttl_days),
            # GSI1 keys for miner+revision queries
            'gsi1_pk': self._make_gsi1_pk(miner_hotkey, model_revision),
            'gsi1_sk': self._make_gsi1_sk(env, task_id),
        }
        
        return await self.put(item)
    
    async def batch_create_tasks(
        self,
        tasks: List[Dict[str, Any]],
        priority: int = 1000,
        ttl_days: int = 7
    ) -> int:
        """Batch create multiple tasks.
        
        Args:
            tasks: List of task dicts with keys:
                - miner_hotkey
                - model_revision
                - model
                - env
                - task_id (integer)
            priority: Default priority for all tasks
            ttl_days: Days until tasks expire
            
        Returns:
            Number of tasks created
        """
        items = []
        for task_info in tasks:
            task_uuid = str(uuid.uuid4())
            created_at = int(time.time())
            status = 'pending'
            
            item = {
                'pk': self._make_pk(task_info['env']),
                'sk': self._make_sk(status, priority, created_at, task_uuid),
                'task_uuid': task_uuid,
                'task_id': task_info['task_id'],
                'miner_hotkey': task_info['miner_hotkey'],
                'model_revision': task_info['model_revision'],
                'model': task_info['model'],
                'env': task_info['env'],
                'status': status,
                'priority': priority,
                'created_at': created_at,
                'assigned_to': None,
                'assigned_at': None,
                'retry_count': 0,
                'max_retries': 3,
                'last_error': None,
                'last_error_code': None,
                'last_failed_at': None,
                'ttl': self.get_ttl(ttl_days),
                'gsi1_pk': self._make_gsi1_pk(task_info['miner_hotkey'], task_info['model_revision']),
                'gsi1_sk': self._make_gsi1_sk(task_info['env'], task_info['task_id']),
            }
            items.append(item)
        
        await self.batch_write(items)
        return len(items)
    
    async def get_task_by_uuid(self, task_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a task by its UUID using GSI (reverse lookup).
        
        Args:
            task_uuid: Task UUID
            
        Returns:
            Task if found, None otherwise
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'task-uuid-index',
            'KeyConditionExpression': 'task_uuid = :task_uuid',
            'ExpressionAttributeValues': {':task_uuid': {'S': task_uuid}},
            'Limit': 1
        }
        
        response = await client.query(**params)
        items = response.get('Items', [])
        
        if not items:
            return None
        
        return self._deserialize(items[0])
    
    async def get_pending_tasks_by_env(
        self,
        env: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get pending tasks for a specific environment.
        
        Tasks are sorted by priority (descending) then by creation time (ascending).
        
        Args:
            env: Environment name
            limit: Maximum number of tasks to return
            
        Returns:
            List of pending tasks
        """
        pk = self._make_pk(env)
        
        # Query tasks with pending status prefix
        # SK format: STATUS#pending#PRIORITY#...
        tasks = await self.query(
            pk=pk,
            sk_prefix='STATUS#pending',
            limit=limit,
            reverse=True  # Higher priority first (lexicographic descending)
        )
        
        return tasks
    
    async def get_next_pending_task(self, env: str) -> Optional[Dict[str, Any]]:
        """Get the next pending task for an environment.
        
        Returns the highest priority, oldest task.
        
        Args:
            env: Environment name
            
        Returns:
            Next task to execute, or None if no pending tasks
        """
        tasks = await self.get_pending_tasks_by_env(env, limit=1)
        return tasks[0] if tasks else None
    
    async def assign_task(
        self,
        task: Dict[str, Any],
        executor_hotkey: str
    ) -> Dict[str, Any]:
        """Assign a task to an executor.
        
        This changes the task status from 'pending' to 'assigned',
        which requires deleting the old item and creating a new one
        (since status is part of the sort key).
        
        Args:
            task: Task dict (from get_next_pending_task)
            executor_hotkey: Executor's hotkey
            
        Returns:
            Updated task
        """
        # Delete old task (with old SK)
        await self.delete(task['pk'], task['sk'])
        
        # Create new task with updated status
        new_status = 'assigned'
        assigned_at = int(time.time())
        
        new_sk = self._make_sk(
            new_status,
            task['priority'],
            task['created_at'],
            task['task_uuid']
        )
        
        task['sk'] = new_sk
        task['status'] = new_status
        task['assigned_to'] = executor_hotkey
        task['assigned_at'] = assigned_at
        
        await self.put(task)
        return task
    
    async def complete_task(self, task: Dict[str, Any]) -> bool:
        """Mark task as completed and delete it from queue.
        
        Args:
            task: Task dict
            
        Returns:
            True if deleted successfully
        """
        return await self.delete(task['pk'], task['sk'])
    
    async def fail_task(
        self,
        task: Dict[str, Any],
        error_message: str,
        error_code: str = 'EXECUTION_ERROR'
    ) -> Dict[str, Any]:
        """Record task failure and handle retry logic.
        
        If retry_count < max_retries, reset status to 'pending'.
        Otherwise, mark as 'failed' permanently.
        
        Args:
            task: Task dict
            error_message: Error description
            error_code: Error classification code
            
        Returns:
            Updated task
        """
        # Delete old task
        await self.delete(task['pk'], task['sk'])
        
        retry_count = task.get('retry_count', 0) + 1
        max_retries = task.get('max_retries', 3)
        
        if retry_count >= max_retries:
            new_status = 'failed'
        else:
            new_status = 'pending'  # Back to pending for retry
        
        new_sk = self._make_sk(
            new_status,
            task['priority'],
            task['created_at'],
            task['task_uuid']
        )
        
        task['sk'] = new_sk
        task['status'] = new_status
        task['retry_count'] = retry_count
        task['last_error'] = error_message
        task['last_error_code'] = error_code
        task['last_failed_at'] = int(time.time())
        task['assigned_to'] = None
        task['assigned_at'] = None
        
        await self.put(task)
        return task
    
    async def get_tasks_by_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all tasks for a specific miner+revision.
        
        Uses GSI1 for efficient querying.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            env: Optional environment filter
            
        Returns:
            List of tasks
        """
        from affine.database.client import get_client
        client = get_client()
        
        gsi1_pk = self._make_gsi1_pk(miner_hotkey, model_revision)
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'miner-revision-index',
            'KeyConditionExpression': 'gsi1_pk = :pk',
            'ExpressionAttributeValues': {':pk': {'S': gsi1_pk}}
        }
        
        if env:
            params['KeyConditionExpression'] = 'gsi1_pk = :pk AND begins_with(gsi1_sk, :sk_prefix)'
            params['ExpressionAttributeValues'][':sk_prefix'] = {'S': f'ENV#{env}'}
        
        response = await client.query(**params)
        return [self._deserialize(item) for item in response.get('Items', [])]
    
    async def get_pending_task_ids_for_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str
    ) -> Set[int]:
        """Get set of task_ids that are already in queue for a miner's env.
        
        Used by task generator to avoid creating duplicate tasks.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            env: Environment name
            
        Returns:
            Set of task_ids (integers)
        """
        tasks = await self.get_tasks_by_miner(miner_hotkey, model_revision, env)
        return {task['task_id'] for task in tasks if task.get('status') in ['pending', 'assigned']}
    
    async def delete_miner_tasks(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: Optional[str] = None
    ) -> int:
        """Delete all tasks for a miner (used when miner is no longer valid).
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            env: Optional environment filter
            
        Returns:
            Number of tasks deleted
        """
        tasks = await self.get_tasks_by_miner(miner_hotkey, model_revision, env)
        
        for task in tasks:
            await self.delete(task['pk'], task['sk'])
        
        return len(tasks)
    
    async def cleanup_invalid_tasks(
        self,
        valid_miners: List[Dict[str, Any]]
    ) -> int:
        """Remove tasks for miners that are no longer valid.
        
        Args:
            valid_miners: List of valid miner dicts with 'hotkey' and 'model_revision'
            
        Returns:
            Number of tasks cleaned up
        """
        # Build set of valid (hotkey, revision) tuples
        valid_set = {
            (m['hotkey'], m['model_revision'])
            for m in valid_miners
        }
        
        # Scan all pending and assigned tasks
        from affine.database.client import get_client
        client = get_client()
        
        # This is a scan operation - use sparingly
        params = {
            'TableName': self.table_name,
            'FilterExpression': '#status IN (:pending, :assigned)',
            'ExpressionAttributeNames': {'#status': 'status'},
            'ExpressionAttributeValues': {
                ':pending': {'S': 'pending'},
                ':assigned': {'S': 'assigned'}
            }
        }
        
        response = await client.scan(**params)
        tasks = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Delete invalid tasks
        deleted_count = 0
        for task in tasks:
            key = (task.get('miner_hotkey'), task.get('model_revision'))
            if key not in valid_set:
                await self.delete(task['pk'], task['sk'])
                deleted_count += 1
        
        return deleted_count
    
    async def get_queue_stats(self, env: str) -> Dict[str, int]:
        """Get statistics about the task queue for an environment.
        
        Args:
            env: Environment name
            
        Returns:
            Dict with counts: pending, assigned, failed
        """
        pk = self._make_pk(env)
        
        # Count by status
        stats = {
            'pending': 0,
            'assigned': 0,
            'failed': 0
        }
        
        for status in stats.keys():
            tasks = await self.query(pk=pk, sk_prefix=f'STATUS#{status}')
            stats[status] = len(tasks)
        
        return stats