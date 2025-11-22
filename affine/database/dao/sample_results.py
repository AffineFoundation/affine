"""
Sample Results DAO

Handles storage and retrieval of sampling results with compression.
"""

import json
import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Tuple
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
            'gsi_partition': 'SAMPLE',  # Fixed partition key for timestamp-index GSI
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
    
    async def get_scoring_samples_batch(
        self,
        miners: List[Dict[str, str]],
        env_ranges: Dict[str, Tuple[int, int]],
        batch_size: int = 30
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Batch query samples for all miners across all environments with concurrent execution.
        
        Args:
            miners: List of miner info dicts with 'hotkey' and 'revision' keys
            env_ranges: Dict mapping env name to (start_id, end_id) tuple
            batch_size: Number of concurrent queries per batch (default: 30)
            
        Returns:
            Dict mapping 'hotkey#revision' to dict of env -> samples list
        """
        from affine.database.client import get_client
        client = get_client()
        
        # Build all query tasks
        tasks = []
        task_metadata = []
        
        for miner in miners:
            hotkey = miner['hotkey']
            revision = miner['revision']
            
            for env, (start_id, end_id) in env_ranges.items():
                if start_id >= end_id:
                    continue
                
                pk = self._make_pk(hotkey, revision, env)
                
                # Build query params with FilterExpression for task_id range
                params = {
                    'TableName': self.table_name,
                    'KeyConditionExpression': 'pk = :pk',
                    'ExpressionAttributeValues': {
                        ':pk': {'S': pk},
                        ':start': {'N': str(start_id)},
                        ':end': {'N': str(end_id)}
                    },
                    'FilterExpression': 'task_id >= :start AND task_id < :end',
                    'ProjectionExpression': 'task_id,score,#ts',
                    'ExpressionAttributeNames': {'#ts': 'timestamp'},
                    'ScanIndexForward': False
                }
                
                tasks.append(client.query(**params))
                task_metadata.append((hotkey, revision, env))
        
        # Execute queries in batches
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            all_results.extend(batch_results)
        
        # Process results and deduplicate
        output = {}
        for (hotkey, revision, env), result in zip(task_metadata, all_results):
            if isinstance(result, Exception):
                continue
            
            items = [self._deserialize(item) for item in result.get('Items', [])]
            
            # Deduplicate: keep latest sample per task_id
            task_samples = {}
            for item in items:
                task_id = int(item['task_id'])
                if task_id not in task_samples or item['timestamp'] > task_samples[task_id]['timestamp']:
                    task_samples[task_id] = item
            
            # Store in output structure
            key = f"{hotkey}#{revision}"
            if key not in output:
                output[key] = {}
            output[key][env] = list(task_samples.values())
        
        return output
    
    async def get_changed_miner_envs_since(
        self,
        since_timestamp: int
    ) -> List[Tuple[str, str, str]]:
        """Detect (hotkey, revision, env) combinations with new samples since timestamp.
        
        Uses timestamp-index GSI for efficient range queries.
        GSI design: gsi_partition='SAMPLE' (HASH) + timestamp (RANGE)
        
        Args:
            since_timestamp: Query samples newer than this timestamp (milliseconds)
            
        Returns:
            List of (hotkey, revision, env) tuples with new samples
        """
        from affine.database.client import get_client
        client = get_client()
        
        # Query timestamp-index GSI with range condition
        params = {
            'TableName': self.table_name,
            'IndexName': 'timestamp-index',
            'KeyConditionExpression': 'gsi_partition = :partition AND #ts > :since',
            'ExpressionAttributeNames': {'#ts': 'timestamp'},
            'ExpressionAttributeValues': {
                ':partition': {'S': 'SAMPLE'},
                ':since': {'N': str(since_timestamp)}
            },
            'ProjectionExpression': 'pk'
        }
        
        # Handle pagination for large result sets
        changed_pks = set()
        
        while True:
            response = await client.query(**params)
            
            for item in response.get('Items', []):
                pk_value = item['pk']['S']
                changed_pks.add(pk_value)
            
            # Check for more pages
            if 'LastEvaluatedKey' not in response:
                break
            params['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
        # Parse PKs to extract (hotkey, revision, env)
        result = []
        for pk in changed_pks:
            # PK format: MINER#{hotkey}#REV#{revision}#ENV#{env}
            parts = pk.split('#')
            if len(parts) >= 6:
                hotkey = parts[1]
                revision = parts[3]
                env = parts[5]
                result.append((hotkey, revision, env))
        
        return result
    
    async def get_scoring_samples_incremental(
        self,
        changed_combos: List[Tuple[str, str, str]],
        env_ranges: Dict[str, Tuple[int, int]],
        since_timestamp: int,
        batch_size: int = 30
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Incrementally query samples for changed miner×env combinations.
        
        Args:
            changed_combos: List of (hotkey, revision, env) tuples with changes
            env_ranges: Dict mapping env name to (start_id, end_id) tuple
            since_timestamp: Only fetch samples newer than this timestamp
            batch_size: Number of concurrent queries per batch
            
        Returns:
            Dict mapping 'hotkey#revision' to dict of env -> samples list
        """
        from affine.database.client import get_client
        client = get_client()
        
        # Build query tasks
        tasks = []
        task_metadata = []
        
        for hotkey, revision, env in changed_combos:
            if env not in env_ranges:
                continue
            
            start_id, end_id = env_ranges[env]
            if start_id >= end_id:
                continue
            
            pk = self._make_pk(hotkey, revision, env)
            
            # Query with both task_id range and timestamp filter
            params = {
                'TableName': self.table_name,
                'KeyConditionExpression': 'pk = :pk',
                'ExpressionAttributeValues': {
                    ':pk': {'S': pk},
                    ':start_id': {'N': str(start_id)},
                    ':end_id': {'N': str(end_id)},
                    ':since_ts': {'N': str(since_timestamp)}
                },
                'FilterExpression': 'task_id >= :start_id AND task_id < :end_id AND #ts > :since_ts',
                'ProjectionExpression': 'task_id,score,#ts',
                'ExpressionAttributeNames': {'#ts': 'timestamp'},
                'ScanIndexForward': False
            }
            
            tasks.append(client.query(**params))
            task_metadata.append((hotkey, revision, env))
        
        # Execute in batches
        all_results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            all_results.extend(batch_results)
        
        # Process results
        output = {}
        for (hotkey, revision, env), result in zip(task_metadata, all_results):
            if isinstance(result, Exception):
                logger.warning(f"[DAO] Query failed for {hotkey[:8]}...#{env}: {result}")
                continue
            
            items = [self._deserialize(item) for item in result.get('Items', [])]
            
            # Deduplicate by task_id
            task_samples = {}
            for item in items:
                task_id = int(item['task_id'])
                if task_id not in task_samples or item['timestamp'] > task_samples[task_id]['timestamp']:
                    task_samples[task_id] = item
            
            # Store in output
            key = f"{hotkey}#{revision}"
            if key not in output:
                output[key] = {}
            output[key][env] = list(task_samples.values())
        
        return output
    
    