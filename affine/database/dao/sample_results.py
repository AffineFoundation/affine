"""
Sample Results DAO

Handles storage and retrieval of sampling results with compression.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class SampleResultsDAO(BaseDAO):
    """DAO for sample_results table.
    
    Stores sampling results with compressed extra data.
    
    Schema Design Philosophy:
    - PK combines the 3 most frequent query dimensions: hotkey + revision + env
    - SK uses task_id for natural ordering
    - uid removed (mutable, should query via bittensor metadata -> hotkey first)
    - GSI for timestamp range queries only
    
    PK: MINER#{hotkey}#REV#{revision}#ENV#{env}
    SK: TASK#{task_id}
    
    The extra field contains conversation and request data, compressed for storage efficiency.
    """
    
    def __init__(self):
        self.table_name = get_table_name("sample_results")
        super().__init__()
    
    def _make_pk(self, miner_hotkey: str, model_revision: str, env: str) -> str:
        """Generate partition key.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name (e.g., affine:sat, agentgym:webshop)
        
        Returns:
            PK string combining hotkey, revision, and env
        """
        return f"MINER#{miner_hotkey}#REV#{model_revision}#ENV#{env}"
    
    def _make_sk(self, task_id: str) -> str:
        """Generate sort key.
        
        Args:
            task_id: Task identifier
        
        Returns:
            SK string with task_id for natural ordering
        """
        return f"TASK#{task_id}"
    
    async def save_sample(
        self,
        miner_hotkey: str,
        model_revision: str,
        model: str,
        env: str,
        task_id: str,
        score: float,
        latency_ms: int,
        extra: Dict[str, Any],
        validator_hotkey: str,
        block_number: int,
        signature: str,
        timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """Save a sampling result.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            model: Model repo/name
            env: Environment name (e.g., affine:sat, agentgym:webshop)
            task_id: Task identifier
            score: Score achieved
            latency_ms: Latency in milliseconds
            extra: Extra data containing conversation and request (will be compressed)
            validator_hotkey: Validator's hotkey
            block_number: Current block number
            signature: Cryptographic signature for verification
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Saved item
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)  # milliseconds
        
        # Compress extra data (contains conversation + request)
        extra_json = json.dumps(extra, separators=(',', ':'))
        extra_compressed = self.compress_data(extra_json)
        
        item = {
            'pk': self._make_pk(miner_hotkey, model_revision, env),
            'sk': self._make_sk(task_id),
            'miner_hotkey': miner_hotkey,
            'model_revision': model_revision,
            'model': model,
            'env': env,
            'task_id': task_id,
            'score': score,
            'latency_ms': latency_ms,
            'timestamp': timestamp,
            'extra_compressed': extra_compressed,
            'validator_hotkey': validator_hotkey,
            'block_number': block_number,
            'signature': signature,
        }
        
        return await self.put(item)
    
    async def get_samples_by_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        limit: Optional[int] = None,
        reverse: bool = True,
        include_extra: bool = True
    ) -> List[Dict[str, Any]]:
        """Get samples for a specific miner, revision, and environment.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name (required, part of PK)
            limit: Maximum number of results
            reverse: If True, return newest first (by task_id)
            include_extra: If True, include and decompress extra field (default: True)
            
        Returns:
            List of samples (extra data decompressed if include_extra=True)
        """
        pk = self._make_pk(miner_hotkey, model_revision, env)
        
        # Build projection expression to exclude extra_compressed if not needed
        from affine.database.client import get_client
        client = get_client()
        
        # Build query parameters
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk',
            'ExpressionAttributeValues': {':pk': {'S': pk}},
            'ScanIndexForward': not reverse
        }
        
        if limit:
            params['Limit'] = limit
        
        # Exclude extra_compressed field if not needed (saves bandwidth and decompression cost)
        if not include_extra:
            params['ProjectionExpression'] = 'pk,sk,miner_hotkey,model_revision,#m,env,task_id,score,latency_ms,#ts,validator_hotkey,block_number,signature'
            params['ExpressionAttributeNames'] = {'#m': 'model', '#ts': 'timestamp'}
        
        # Execute query
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Decompress extra data if included
        if include_extra:
            for item in items:
                if 'extra_compressed' in item:
                    compressed = item['extra_compressed']
                    extra_json = self.decompress_data(compressed)
                    item['extra'] = json.loads(extra_json)
                    del item['extra_compressed']
        
        return items
    
    async def get_samples_by_miner_all_envs(
        self,
        miner_hotkey: str,
        model_revision: str,
        envs: List[str],
        limit_per_env: Optional[int] = None,
        include_extra: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get samples for a miner across multiple environments.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            envs: List of environment names to query
            limit_per_env: Maximum number of results per environment
            include_extra: If True, include and decompress extra field (default: True)
            
        Returns:
            Dict mapping env -> list of samples
        """
        results = {}
        for env in envs:
            samples = await self.get_samples_by_miner(
                miner_hotkey=miner_hotkey,
                model_revision=model_revision,
                env=env,
                limit=limit_per_env,
                include_extra=include_extra
            )
            results[env] = samples
        
        return results
    
    async def get_samples_by_timestamp_range(
        self,
        start_timestamp: int,
        end_timestamp: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get samples by timestamp range.
        
        Uses timestamp-index GSI for time-based queries.
        
        Args:
            start_timestamp: Start timestamp in milliseconds (inclusive)
            end_timestamp: End timestamp in milliseconds (inclusive)
            limit: Maximum number of results
            
        Returns:
            List of samples
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'timestamp-index',
            'KeyConditionExpression': '#ts BETWEEN :start AND :end',
            'ExpressionAttributeNames': {'#ts': 'timestamp'},
            'ExpressionAttributeValues': {
                ':start': {'N': str(start_timestamp)},
                ':end': {'N': str(end_timestamp)}
            }
        }
        
        if limit:
            params['Limit'] = limit
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Decompress extra data
        for item in items:
            if 'extra_compressed' in item:
                compressed = item['extra_compressed']
                extra_json = self.decompress_data(compressed)
                item['extra'] = json.loads(extra_json)
                del item['extra_compressed']
        
        return items
    
    async def delete_samples_before(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        cutoff_timestamp: int
    ) -> int:
        """Delete samples older than cutoff timestamp.
        
        Used by cleanup scripts for non-protected miners.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            cutoff_timestamp: Delete samples before this timestamp (milliseconds)
            
        Returns:
            Number of samples deleted
        """
        pk = self._make_pk(miner_hotkey, model_revision, env)
        
        # Query all samples for this PK
        items = await self.query(pk=pk, reverse=False)
        
        deleted_count = 0
        for item in items:
            if item['timestamp'] < cutoff_timestamp:
                await self.delete(item['pk'], item['sk'])
                deleted_count += 1
        
        return deleted_count
    
    async def get_completed_task_ids(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str
    ) -> set:
        """Get set of completed task_ids for a miner's env.
        
        Used by task generator to determine which dataset indices are missing.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            
        Returns:
            Set of completed task_ids (integers representing dataset indices)
        """
        # Query all samples without decompressing extra data (for performance)
        samples = await self.get_samples_by_miner(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=env,
            include_extra=False
        )
        
        # Extract task_ids, converting from string format "42" to int 42
        task_ids = set()
        for sample in samples:
            task_id = sample.get('task_id')
            if task_id is not None:
                # Handle both string and int formats
                if isinstance(task_id, str):
                    try:
                        task_ids.add(int(task_id))
                    except ValueError:
                        # Skip non-integer task_ids (old format)
                        pass
                else:
                    task_ids.add(int(task_id))
        
        return task_ids
    
    async def get_samples_with_completion_status(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        dataset_length: int,
        deduplicate_by_task_id: bool = True,
        include_extra: bool = False,
        task_id_start: Optional[int] = None,
        task_id_end: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get samples with completion status for scoring.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            dataset_length: Total number of tasks in the dataset
            deduplicate_by_task_id: If True, keep only latest sample per task_id
            include_extra: If True, include and decompress extra field
            task_id_start: Optional start of task ID range (inclusive, overrides dataset_length)
            task_id_end: Optional end of task ID range (exclusive, overrides dataset_length)
            
        Returns:
            Dict containing:
                - samples: List of samples (optionally deduplicated)
                - is_complete: Boolean indicating if all tasks are sampled
                - completed_count: Number of unique task_ids completed
                - total_count: Total number of tasks in dataset
                - missing_task_ids: List of missing task_ids (if incomplete)
        """
        samples = await self.get_samples_by_miner(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=env,
            include_extra=include_extra
        )
        
        # Determine expected task ID range
        if task_id_start is not None and task_id_end is not None:
            # Use explicit range (for dataset expansion transitions)
            expected_task_ids = set(range(task_id_start, task_id_end))
        else:
            # Default: 0 to dataset_length
            expected_task_ids = set(range(dataset_length))
        
        if deduplicate_by_task_id:
            # Keep only the latest sample for each task_id (by timestamp)
            task_id_samples = {}
            for sample in samples:
                task_id = sample.get('task_id')
                if task_id is not None:
                    # Convert to int for comparison
                    if isinstance(task_id, str):
                        try:
                            task_id_int = int(task_id)
                        except ValueError:
                            continue
                    else:
                        task_id_int = int(task_id)
                    
                    # Only include task_ids within expected range
                    if task_id_int not in expected_task_ids:
                        continue
                    
                    # Keep the latest (newest) sample
                    if task_id_int not in task_id_samples:
                        task_id_samples[task_id_int] = sample
                    else:
                        existing_ts = task_id_samples[task_id_int].get('timestamp', 0)
                        new_ts = sample.get('timestamp', 0)
                        if new_ts > existing_ts:
                            task_id_samples[task_id_int] = sample
            
            samples = list(task_id_samples.values())
            completed_task_ids = set(task_id_samples.keys())
        else:
            # Extract all unique task_ids within expected range
            completed_task_ids = set()
            for sample in samples:
                task_id = sample.get('task_id')
                if task_id is not None:
                    if isinstance(task_id, str):
                        try:
                            task_id_int = int(task_id)
                        except ValueError:
                            continue
                    else:
                        task_id_int = int(task_id)
                    
                    # Only count task_ids within expected range
                    if task_id_int in expected_task_ids:
                        completed_task_ids.add(task_id_int)
        
        # Calculate completion status
        missing_task_ids = expected_task_ids - completed_task_ids
        is_complete = len(missing_task_ids) == 0
        
        return {
            'samples': samples,
            'is_complete': is_complete,
            'completed_count': len(completed_task_ids),
            'total_count': len(expected_task_ids),
            'missing_task_ids': sorted(list(missing_task_ids)) if not is_complete else []
        }