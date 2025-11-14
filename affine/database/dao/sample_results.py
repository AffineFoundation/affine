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
    
    Stores sampling results with compressed conversation data.
    PK: MINER#{hotkey}#REV#{revision}
    SK: ENV#{env}#TIME#{timestamp}#ID#{uuid}
    """
    
    def __init__(self):
        self.table_name = get_table_name("sample_results")
        super().__init__()
    
    def _make_pk(self, miner_hotkey: str, model_revision: str) -> str:
        """Generate partition key."""
        return f"MINER#{miner_hotkey}#REV#{model_revision}"
    
    def _make_sk(self, env: str, timestamp: int, sample_id: str) -> str:
        """Generate sort key."""
        return f"ENV#{env}#TIME#{timestamp:016d}#ID#{sample_id}"
    
    async def save_sample(
        self,
        miner_hotkey: str,
        model_revision: str,
        model_hash: str,
        uid: int,
        env: str,
        subset: str,
        dataset_index: int,
        task_id: str,
        score: float,
        success: bool,
        latency_ms: int,
        conversation: Dict[str, Any],
        validator_hotkey: str,
        block_number: int,
        signature: str,
        timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """Save a sampling result.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            model_hash: Model hash for validation
            uid: Miner UID
            env: Environment name (L3-L8)
            subset: Dataset subset
            dataset_index: Index in dataset
            task_id: Task ID
            score: Score achieved
            success: Whether sampling succeeded
            latency_ms: Latency in milliseconds
            conversation: Conversation data (will be compressed)
            validator_hotkey: Validator's hotkey
            block_number: Current block number
            signature: Cryptographic signature
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            Saved item
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)  # milliseconds
        
        sample_id = str(uuid.uuid4())
        
        # Compress conversation data
        conversation_json = json.dumps(conversation, separators=(',', ':'))
        conversation_compressed = self.compress_data(conversation_json)
        
        item = {
            'pk': self._make_pk(miner_hotkey, model_revision),
            'sk': self._make_sk(env, timestamp, sample_id),
            'sample_id': sample_id,
            'miner_hotkey': miner_hotkey,
            'model_revision': model_revision,
            'model_hash': model_hash,
            'uid': uid,
            'env': env,
            'subset': subset,
            'dataset_index': dataset_index,
            'task_id': task_id,
            'score': score,
            'success': success,
            'latency_ms': latency_ms,
            'timestamp': timestamp,
            'conversation_compressed': conversation_compressed,
            'validator_hotkey': validator_hotkey,
            'block_number': block_number,
            'signature': signature,
        }
        
        return await self.put(item)
    
    async def get_samples_by_miner(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: Optional[str] = None,
        limit: Optional[int] = None,
        reverse: bool = True
    ) -> List[Dict[str, Any]]:
        """Get samples for a specific miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            env: Optional environment filter
            limit: Maximum number of results
            reverse: If True, return newest first
            
        Returns:
            List of samples (conversation data decompressed)
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        sk_prefix = f"ENV#{env}#" if env else None
        
        items = await self.query(
            pk=pk,
            sk_prefix=sk_prefix,
            limit=limit,
            reverse=reverse
        )
        
        # Decompress conversation data
        for item in items:
            if 'conversation_compressed' in item:
                compressed = item['conversation_compressed']
                conversation_json = self.decompress_data(compressed)
                item['conversation'] = json.loads(conversation_json)
                del item['conversation_compressed']
        
        return items
    
    async def get_samples_by_env(
        self,
        env: str,
        start_timestamp: int,
        end_timestamp: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get samples by environment and time range.
        
        Uses env-timestamp-index GSI.
        
        Args:
            env: Environment name
            start_timestamp: Start timestamp (inclusive)
            end_timestamp: End timestamp (inclusive)
            limit: Maximum number of results
            
        Returns:
            List of samples
        """
        client = self.get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'env-timestamp-index',
            'KeyConditionExpression': 'env = :env AND #ts BETWEEN :start AND :end',
            'ExpressionAttributeNames': {'#ts': 'timestamp'},
            'ExpressionAttributeValues': {
                ':env': {'S': env},
                ':start': {'N': str(start_timestamp)},
                ':end': {'N': str(end_timestamp)}
            }
        }
        
        if limit:
            params['Limit'] = limit
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Decompress conversation data
        for item in items:
            if 'conversation_compressed' in item:
                compressed = item['conversation_compressed']
                conversation_json = self.decompress_data(compressed)
                item['conversation'] = json.loads(conversation_json)
                del item['conversation_compressed']
        
        return items
    
    async def delete_samples_before(
        self,
        miner_hotkey: str,
        model_revision: str,
        cutoff_timestamp: int
    ) -> int:
        """Delete samples older than cutoff timestamp.
        
        Used by cleanup scripts for non-protected miners.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision
            cutoff_timestamp: Delete samples before this timestamp
            
        Returns:
            Number of samples deleted
        """
        pk = self._make_pk(miner_hotkey, model_revision)
        
        # Query samples before cutoff
        items = await self.query(pk=pk, reverse=False)
        
        deleted_count = 0
        for item in items:
            if item['timestamp'] < cutoff_timestamp:
                await self.delete(item['pk'], item['sk'])
                deleted_count += 1
            else:
                # Items are ordered by timestamp, so we can stop
                break
        
        return deleted_count