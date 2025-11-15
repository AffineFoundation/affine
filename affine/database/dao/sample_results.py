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
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """Get samples for a specific miner, revision, and environment.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name (required, part of PK)
            limit: Maximum number of results
            reverse: If True, return newest first (by task_id)
            
        Returns:
            List of samples (extra data decompressed)
        """
        pk = self._make_pk(miner_hotkey, model_revision, env)
        
        items = await self.query(
            pk=pk,
            limit=limit,
            reverse=reverse
        )
        
        # Decompress extra data
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
        limit_per_env: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get samples for a miner across multiple environments.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            envs: List of environment names to query
            limit_per_env: Maximum number of results per environment
            
        Returns:
            Dict mapping env -> list of samples
        """
        results = {}
        for env in envs:
            samples = await self.get_samples_by_miner(
                miner_hotkey=miner_hotkey,
                model_revision=model_revision,
                env=env,
                limit=limit_per_env
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