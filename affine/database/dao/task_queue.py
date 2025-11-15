"""
Task Queue DAO

Manages sampling tasks with status tracking and error handling.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class TaskQueueDAO(BaseDAO):
    """DAO for task_queue table.
    
    Manages sampling tasks for miners.
    PK: MINER#{hotkey}#REV#{revision}
    SK: TASK#{task_id}
    """
    
    def __init__(self):
        self.table_name = get_table_name("task_queue")
        super().__init__()
    
    def _make_pk(self, miner_hotkey: str, model_revision: str) -> str:
        """Generate partition key."""
        return f"MINER#{miner_hotkey}#REV#{model_revision}"
    
    def _make_sk(self, task_id: str) -> str:
        """Generate sort key."""
        return f"TASK#{task_id}"
    
    async def create_task(
        self,
        miner_hotkey: str,
        model_revision: str,
        model: str,
        env: str,
        validator_hotkey: str,
        priority: int = 0,
        ttl_days: int = 7
    ) -> Dict[str, Any]:
        """Create a new sampling task.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            model: Model repo/name for pre-execution validation
            env: Environment name (L3-L8)
            validator_hotkey: Validator's hotkey (currently owner_hotkey)
            priority: Task priority (higher = more urgent)
            ttl_days: Days until task expires
            
        Returns:
            Created task
        """
        # Generate time-based UUID for ordering
        task_id = f"{int(time.time()*1000)}-{uuid.uuid4()}"
        
        created_at = int(time.time())
        
        item = {
            'pk': self._make_pk(miner_hotkey, model_revision),
            'sk': self._make_sk(task_id),
            'task_id': task_id,
            'miner_hotkey': miner_hotkey,
            'model_revision': model_revision,
            'model': model,
            'env': env,
            'status': 'pending',
            'priority': priority,
            'created_at': created_at,
            'started_at': None,
            'completed_at': None,
            'error_count': 0,
            'last_error': None,
            'validator_hotkey': validator_hotkey,
            'ttl': self.get_ttl(ttl_days),
        }
        
        return await self.put(item)
    
    async def get_task(self, miner_hotkey: str, model_revision: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific task.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        sk = self._make_sk(task_id)
        
        return await self.get(pk, sk)
    
    async def get_pending_tasks(
        self,
        limit: Optional[int] = None,
        priority_order: bool = True
    ) -> List[Dict[str, Any]]:
        """Get pending tasks across all miners.
        
        Uses status-created-index GSI.
        
        Args:
            limit: Maximum number of tasks to return
            priority_order: If True, sort by priority (high to low)
            
        Returns:
            List of pending tasks
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'status-created-index',
            'KeyConditionExpression': '#status = :status',
            'ExpressionAttributeNames': {'#status': 'status'},
            'ExpressionAttributeValues': {':status': {'S': 'pending'}}
        }
        
        if limit:
            params['Limit'] = limit
        
        response = await client.query(**params)
        tasks = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Sort by priority if requested
        if priority_order:
            tasks.sort(key=lambda t: t.get('priority', 0), reverse=True)
        
        return tasks
    
    async def start_task(self, miner_hotkey: str, model_revision: str, task_id: str) -> bool:
        """Mark task as started.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            task_id: Task ID
            
        Returns:
            True if updated successfully
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        sk = self._make_sk(task_id)
        
        task = await self.get(pk, sk)
        if not task:
            return False
        
        task['status'] = 'running'
        task['started_at'] = int(time.time())
        
        await self.put(task)
        return True
    
    async def complete_task(self, miner_hotkey: str, model_revision: str, task_id: str) -> bool:
        """Mark task as completed and delete it.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            task_id: Task ID
            
        Returns:
            True if deleted successfully
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        sk = self._make_sk(task_id)
        
        return await self.delete(pk, sk)
    
    async def fail_task(
        self,
        miner_hotkey: str,
        model_revision: str,
        task_id: str,
        error_message: str
    ) -> bool:
        """Record task failure and increment error count.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            task_id: Task ID
            error_message: Error description
            
        Returns:
            True if updated successfully
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        sk = self._make_sk(task_id)
        
        task = await self.get(pk, sk)
        if not task:
            return False
        
        task['status'] = 'failed'
        task['error_count'] = task.get('error_count', 0) + 1
        task['last_error'] = error_message
        task['completed_at'] = int(time.time())
        
        await self.put(task)
        return True
    
    async def get_tasks_by_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all tasks for a specific miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List of tasks
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        
        tasks = await self.query(pk=pk, limit=limit, reverse=True)
        
        # Filter by status if specified
        if status:
            tasks = [t for t in tasks if t.get('status') == status]
        
        return tasks
    
    async def delete_miner_tasks(self, miner_hotkey: str, model_revision: str) -> int:
        """Delete all tasks for a miner (used when miner state changes).
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            
        Returns:
            Number of tasks deleted
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        
        tasks = await self.query(pk=pk)
        
        for task in tasks:
            await self.delete(task['pk'], task['sk'])
        
        return len(tasks)