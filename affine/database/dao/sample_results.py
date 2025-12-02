"""
Sample Results DAO

Handles storage and retrieval of sampling results with compression.
"""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client
from affine.core.setup import logger


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
        
        # Ensure task_id is integer for proper range queries
        task_id_int = int(task_id) if not isinstance(task_id, int) else task_id
        
        # Compress extra data (contains conversation + request)
        extra_json = json.dumps(extra, separators=(',', ':'))
        extra_compressed = self.compress_data(extra_json)
        
        item = {
            'pk': self._make_pk(miner_hotkey, model_revision, env),
            'sk': self._make_sk(str(task_id_int)),
            'miner_hotkey': miner_hotkey,
            'model_revision': model_revision,
            'model': model,
            'env': env,
            'task_id': task_id_int,  # Store as integer
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
    
    async def get_sample_by_task_id(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        task_id: str,
        include_extra: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get a specific sample by its full key (PK + SK).
        
        This is the most efficient way to retrieve a single sample.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            task_id: Task identifier
            include_extra: If True, include and decompress extra field
            
        Returns:
            Sample dict or None if not found
        """
        pk = self._make_pk(miner_hotkey, model_revision, env)
        sk = self._make_sk(task_id)
        
        # Use get_item for O(1) direct key access
        item = await self.get(pk, sk)
        
        if not item:
            return None
        
        # Decompress extra data if needed
        if include_extra and 'extra_compressed' in item:
            compressed = item['extra_compressed']
            extra_json = self.decompress_data(compressed)
            item['extra'] = json.loads(extra_json)
            del item['extra_compressed']
        
        return item

    
    def _parse_task_id(self, task_id_field: Dict[str, Any]) -> Optional[int]:
        """Parse task_id from DynamoDB field format.
        
        Args:
            task_id_field: DynamoDB field dict with type indicators
            
        Returns:
            Parsed integer or None if parsing fails
        """
        for type_key in ('N', 'S'):
            if type_key in task_id_field:
                try:
                    return int(task_id_field[type_key])
                except (ValueError, TypeError):
                    pass
        return None
    
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
        pk = self._make_pk(miner_hotkey, model_revision, env)
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk',
            'ExpressionAttributeValues': {':pk': {'S': pk}},
            'ProjectionExpression': 'task_id'
        }
        
        items = await self._query_all_pages(get_client(), params)
        task_ids = set()
        
        for item in items:
            task_id = self._parse_task_id(item.get('task_id', {}))
            if task_id is not None:
                task_ids.add(task_id)
        
        return task_ids
    
    async def _query_all_pages(
        self,
        client,
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query DynamoDB with automatic pagination handling.
        
        Args:
            client: DynamoDB client
            params: Query parameters
            
        Returns:
            List of all items across all pages
        """
        all_items = []
        last_key = None
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.query(**params)
            items = response.get('Items', [])
            all_items.extend(items)
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        return all_items
    
    async def get_scoring_samples_batch(
        self,
        miners: List[Dict[str, str]],
        env_ranges: Dict[str, Tuple[int, int]],
        batch_size: int = 30
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Batch query samples for all miners across all environments with concurrent execution.
        
        Uses DynamoDB FilterExpression for efficient server-side filtering by task_id range.
        
        Args:
            miners: List of miner info dicts with 'hotkey' and 'revision' keys
            env_ranges: Dict mapping env name to (start_id, end_id) tuple
            batch_size: Number of concurrent queries per batch (default: 30)
            
        Returns:
            Dict mapping 'hotkey#revision' to dict of env -> samples list
        """
        client = get_client()
        query_coros = []
        query_metadata = []
        
        for miner in miners:
            hotkey = miner['hotkey']
            revision = miner['revision']
            
            for env, (start_id, end_id) in env_ranges.items():
                if start_id >= end_id:
                    continue
                
                pk = self._make_pk(hotkey, revision, env)
                
                # Use FilterExpression for server-side task_id range filtering
                params = {
                    'TableName': self.table_name,
                    'KeyConditionExpression': 'pk = :pk',
                    'FilterExpression': 'task_id >= :start_id AND task_id < :end_id',
                    'ExpressionAttributeValues': {
                        ':pk': {'S': pk},
                        ':start_id': {'N': str(start_id)},
                        ':end_id': {'N': str(end_id)}
                    },
                    'ProjectionExpression': 'task_id,score,#ts',
                    'ExpressionAttributeNames': {'#ts': 'timestamp'},
                    'ScanIndexForward': False
                }
                
                query_coros.append(self._query_all_pages(client, params))
                query_metadata.append((hotkey, revision, env))
        
        # Execute queries in batches
        all_results = []
        for i in range(0, len(query_coros), batch_size):
            batch = query_coros[i:i+batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            all_results.extend(batch_results)
        
        # Process results (no need for client-side filtering now)
        output = {}
        for metadata, result in zip(query_metadata, all_results):
            hotkey, revision, env = metadata
            
            if isinstance(result, Exception):
                logger.error(f"Query failed for {hotkey[:8]}...#{env}: {result}")
                continue
            
            items = [self._deserialize(item) for item in result]
            
            key = f"{hotkey}#{revision}"
            if key not in output:
                output[key] = {}
            output[key][env] = items
        
        return output
    
    async def delete_samples_by_task_range(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        start_task_id: int,
        end_task_id: int
    ) -> int:
        """Delete samples within a task_id range for a specific miner and environment.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            start_task_id: Start of task_id range (inclusive)
            end_task_id: End of task_id range (exclusive)
            
        Returns:
            Number of samples deleted
        """
        client = get_client()
        pk = self._make_pk(miner_hotkey, model_revision, env)
        
        # Query all samples in the range
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk',
            'FilterExpression': 'task_id >= :start_id AND task_id < :end_id',
            'ExpressionAttributeValues': {
                ':pk': {'S': pk},
                ':start_id': {'N': str(start_task_id)},
                ':end_id': {'N': str(end_task_id)}
            },
            'ProjectionExpression': 'pk, sk'
        }
        
        items = await self._query_all_pages(client, params)
        
        if not items:
            logger.info(f"No samples found in range [{start_task_id}, {end_task_id})")
            return 0
        
        # Batch delete items
        deleted_count = 0
        batch_size = 25
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            delete_requests = [
                {
                    'DeleteRequest': {
                        'Key': {
                            'pk': item['pk'],
                            'sk': item['sk']
                        }
                    }
                }
                for item in batch
            ]
            
            try:
                await client.batch_write_item(
                    RequestItems={
                        self.table_name: delete_requests
                    }
                )
                deleted_count += len(batch)
            except Exception as e:
                logger.error(f"Batch delete failed: {e}")
        
        logger.info(f"Deleted {deleted_count} samples in range [{start_task_id}, {end_task_id})")
        return deleted_count
    
    async def delete_samples_with_empty_conversation(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str
    ) -> int:
        """Delete samples where extra.conversation is empty.
        
        Args:
            miner_hotkey: Miner's hotkey
            model_revision: Model revision hash
            env: Environment name
            
        Returns:
            Number of samples deleted
        """
        client = get_client()
        pk = self._make_pk(miner_hotkey, model_revision, env)
        
        # Query all samples for this miner+env
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': 'pk = :pk',
            'ExpressionAttributeValues': {
                ':pk': {'S': pk}
            }
        }
        
        all_items = await self._query_all_pages(client, params)
        
        if not all_items:
            logger.info(f"No samples found for deletion")
            return 0
        
        # Filter items with empty conversation
        items_to_delete = []
        for item_raw in all_items:
            item = self._deserialize(item_raw)
            
            # Decompress and check extra.conversation and extra.request.max_round
            if 'extra_compressed' in item:
                try:
                    extra_json = self.decompress_data(item['extra_compressed'])
                    extra = json.loads(extra_json)
                    conversation = extra.get('conversation', [])
                    
                    # Check deletion conditions
                    should_delete = False
                    
                    # Condition 1: Empty conversation
                    if not conversation or len(conversation) == 0:
                        should_delete = True
                    
                    # Condition 2: max_round = 10 (only if field exists)
                    if 'request' in extra and isinstance(extra['request'], dict):
                        max_round = extra['request'].get('max_round')
                        if max_round is not None and max_round == 10:
                            should_delete = True
                    
                    if should_delete:
                        items_to_delete.append({
                            'pk': item_raw['pk'],
                            'sk': item_raw['sk']
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse extra for item {item.get('task_id')}: {e}")
        
        if not items_to_delete:
            logger.info(f"No samples with empty conversation found")
            return 0
        
        logger.info(f"Found {len(items_to_delete)} samples with empty conversation")
        
        # Batch delete items
        deleted_count = 0
        batch_size = 25
        
        for i in range(0, len(items_to_delete), batch_size):
            batch = items_to_delete[i:i + batch_size]
            
            delete_requests = [
                {
                    'DeleteRequest': {
                        'Key': {
                            'pk': item['pk'],
                            'sk': item['sk']
                        }
                    }
                }
                for item in batch
            ]
            
            try:
                await client.batch_write_item(
                    RequestItems={
                        self.table_name: delete_requests
                    }
                )
                deleted_count += len(batch)
            except Exception as e:
                logger.error(f"Batch delete failed: {e}")
        
        logger.info(f"Deleted {deleted_count} samples with empty conversation")
        return deleted_count
    
    async def delete_all_samples_with_empty_conversation(self) -> int:
        """Delete all samples with empty conversation across the entire database.
        
        Performs a full table scan using GSI and streams deletion with progress logging.
        
        Returns:
            Number of samples deleted
        """
        client = get_client()
        
        # Use GSI timestamp-index for full table scan
        params = {
            'TableName': self.table_name,
            'IndexName': 'timestamp-index',
            'KeyConditionExpression': 'gsi_partition = :partition',
            'ExpressionAttributeValues': {
                ':partition': {'S': 'SAMPLE'}
            }
        }
        
        # Stream scan with pagination
        scanned_count = 0
        items_to_delete = []
        last_key = None
        
        logger.info("Starting full table scan for samples with empty conversation...")
        print("Scanning database...")
        
        while True:
            if last_key:
                params['ExclusiveStartKey'] = last_key
            
            response = await client.query(**params)
            items = response.get('Items', [])
            
            # Process batch
            for item_raw in items:
                scanned_count += 1
                item = self._deserialize(item_raw)
                
                # Decompress and check extra.conversation and extra.request.max_round
                if 'extra_compressed' in item:
                    try:
                        extra_json = self.decompress_data(item['extra_compressed'])
                        extra = json.loads(extra_json)
                        conversation = extra.get('conversation', [])
                        
                        # Check deletion conditions
                        should_delete = False
                        
                        # Condition 1: Empty conversation
                        if not conversation or len(conversation) == 0:
                            should_delete = True
                        
                        # Condition 2: max_round = 10 (only if field exists)
                        if 'request' in extra and isinstance(extra['request'], dict):
                            max_round = extra['request'].get('max_round')
                            if max_round is not None and max_round == 10:
                                should_delete = True
                        
                        if should_delete:
                            items_to_delete.append({
                                'pk': item_raw['pk'],
                                'sk': item_raw['sk']
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse extra for item {item.get('task_id')}: {e}")
                
                # Log progress every 1000 items scanned
                if scanned_count % 1000 == 0:
                    print(f"Scanned {scanned_count} samples, found {len(items_to_delete)} invalid samples...")
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
        
        print(f"\nScan complete: Scanned {scanned_count} samples, found {len(items_to_delete)} invalid samples")
        
        if not items_to_delete:
            logger.info("No samples with empty conversation found")
            return 0
        
        # Stream deletion with progress logging
        deleted_count = 0
        batch_size = 25
        total_batches = (len(items_to_delete) + batch_size - 1) // batch_size
        
        print(f"Starting deletion in {total_batches} batches...")
        
        for i in range(0, len(items_to_delete), batch_size):
            batch = items_to_delete[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            delete_requests = [
                {
                    'DeleteRequest': {
                        'Key': {
                            'pk': item['pk'],
                            'sk': item['sk']
                        }
                    }
                }
                for item in batch
            ]
            
            try:
                await client.batch_write_item(
                    RequestItems={
                        self.table_name: delete_requests
                    }
                )
                deleted_count += len(batch)
                print(f"Batch {batch_num}/{total_batches}: Deleted {len(batch)} samples (total: {deleted_count})")
            except Exception as e:
                logger.error(f"Batch delete failed at batch {batch_num}: {e}")
        
        logger.info(f"Successfully deleted {deleted_count} samples with empty conversation")
        return deleted_count

