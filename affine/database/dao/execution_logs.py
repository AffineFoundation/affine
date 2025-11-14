"""
Execution Logs DAO

Tracks task execution history with automatic TTL cleanup.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class ExecutionLogsDAO(BaseDAO):
    """DAO for execution_logs table.
    
    Stores execution history with automatic 7-day expiration.
    PK: MINER#{hotkey}
    SK: TIME#{timestamp}#ID#{uuid}
    """
    
    def __init__(self):
        self.table_name = get_table_name("execution_logs")
        super().__init__()
    
    def _make_pk(self, miner_hotkey: str) -> str:
        """Generate partition key."""
        return f"MINER#{miner_hotkey}"
    
    def _make_sk(self, timestamp: int, log_id: str) -> str:
        """Generate sort key."""
        return f"TIME#{timestamp:016d}#ID#{log_id}"
    
    async def log_execution(
        self,
        miner_hotkey: str,
        uid: int,
        task_id: str,
        status: str,
        env: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        execution_time_ms: int = 0,
        timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """Log a task execution.
        
        Args:
            miner_hotkey: Miner's hotkey
            uid: Miner UID
            task_id: Task ID
            status: Execution status (success/failed)
            env: Environment name
            error_type: Optional error type
            error_message: Optional error message
            execution_time_ms: Execution time in milliseconds
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Created log entry
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        log_id = str(uuid.uuid4())
        
        item = {
            'pk': self._make_pk(miner_hotkey),
            'sk': self._make_sk(timestamp, log_id),
            'log_id': log_id,
            'miner_hotkey': miner_hotkey,
            'uid': uid,
            'task_id': task_id,
            'status': status,
            'env': env,
            'error_type': error_type,
            'error_message': error_message,
            'execution_time_ms': execution_time_ms,
            'timestamp': timestamp,
            'ttl': self.get_ttl(7),  # 7 days
        }
        
        return await self.put(item)
    
    async def get_recent_logs(
        self,
        miner_hotkey: str,
        limit: int = 1000,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent logs for a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            limit: Maximum number of logs (default 1000)
            status: Optional status filter
            
        Returns:
            List of log entries (newest first)
        """
        pk = self._make_pk(miner_hotkey)
        
        logs = await self.query(pk=pk, limit=limit, reverse=True)
        
        # Filter by status if specified
        if status:
            logs = [log for log in logs if log.get('status') == status]
        
        return logs
    
    async def check_consecutive_errors(
        self,
        miner_hotkey: str,
        threshold: int = 10
    ) -> bool:
        """Check if miner has consecutive errors exceeding threshold.
        
        Args:
            miner_hotkey: Miner's hotkey
            threshold: Number of consecutive errors to trigger pause
            
        Returns:
            True if consecutive errors >= threshold
        """
        logs = await self.get_recent_logs(miner_hotkey, limit=threshold)
        
        if len(logs) < threshold:
            return False
        
        # Check if all recent logs are failures
        recent_logs = logs[:threshold]
        all_failed = all(log.get('status') == 'failed' for log in recent_logs)
        
        return all_failed
    
    async def get_error_summary(
        self,
        miner_hotkey: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent error details for a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            limit: Number of recent errors to return
            
        Returns:
            List of error details
        """
        logs = await self.get_recent_logs(miner_hotkey, limit=100, status='failed')
        
        # Return only the most recent errors with full details
        return [{
            'timestamp': log['timestamp'],
            'env': log['env'],
            'error_type': log.get('error_type'),
            'error_message': log.get('error_message'),
            'task_id': log.get('task_id')
        } for log in logs[:limit]]
    
    async def get_execution_stats(
        self,
        miner_hotkey: str,
        time_window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Get execution statistics for a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            time_window_seconds: Time window to analyze (default 1 hour)
            
        Returns:
            Statistics including success/failure counts by environment
        """
        cutoff = int(time.time()) - time_window_seconds
        pk = self._make_pk(miner_hotkey)
        
        logs = await self.query(pk=pk, limit=10000, reverse=True)
        
        # Filter logs within time window
        recent_logs = [log for log in logs if log['timestamp'] >= cutoff]
        
        # Calculate statistics
        stats = {
            'total_executions': len(recent_logs),
            'success_count': 0,
            'failure_count': 0,
            'by_env': {},
            'avg_execution_time_ms': 0
        }
        
        total_time = 0
        
        for log in recent_logs:
            env = log.get('env', 'unknown')
            status = log.get('status', 'unknown')
            
            if status == 'success':
                stats['success_count'] += 1
            elif status == 'failed':
                stats['failure_count'] += 1
            
            # Track by environment
            if env not in stats['by_env']:
                stats['by_env'][env] = {'success': 0, 'failed': 0}
            
            stats['by_env'][env][status] = stats['by_env'][env].get(status, 0) + 1
            
            # Track execution time
            total_time += log.get('execution_time_ms', 0)
        
        # Calculate average execution time
        if len(recent_logs) > 0:
            stats['avg_execution_time_ms'] = total_time // len(recent_logs)
        
        return stats