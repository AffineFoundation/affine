"""
Miners DAO

Manages miner validation state and anti-plagiarism tracking.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


logger = logging.getLogger(__name__)


class MinersDAO(BaseDAO):
    """DAO for miners table.
    
    Schema Design:
    - PK: UID#{uid} - partition by UID for fast lookups
    - SK: BLOCK#{block_number}#HOTKEY#{hotkey} - track state changes over time
    - GSI1: is-valid-index for querying valid/invalid miners
    """
    
    def __init__(self):
        self.table_name = get_table_name("miners")
        super().__init__()
    
    def _make_pk(self, uid: int) -> str:
        """Generate partition key based on UID."""
        return f"UID#{uid}"
    
    def _make_sk(self, block_number: int, hotkey: str) -> str:
        """Generate sort key with block number and hotkey."""
        return f"BLOCK#{block_number}#HOTKEY#{hotkey}"
    
    async def save_miner(
        self,
        uid: int,
        hotkey: str,
        model: str,
        revision: str,
        chute_id: str,
        chute_slug: str,
        model_hash: str,
        chute_status: str,
        is_valid: bool,
        invalid_reason: Optional[str],
        block_number: int,
        first_block: int,
        ttl_days: int = 30
    ) -> Dict[str, Any]:
        """Save or update miner validation state.
        
        Args:
            uid: Miner UID (0-255)
            hotkey: Miner's SS58 hotkey
            model: HuggingFace model repo
            revision: Git commit hash
            chute_id: Chutes deployment ID
            chute_slug: Chutes URL slug
            model_hash: SHA256 hash of all model weights
            chute_status: "hot" or "cold"
            is_valid: Overall validation result (boolean)
            invalid_reason: Reason if invalid (null if valid)
            block_number: Block when this record was created
            first_block: Block when miner first committed
            ttl_days: Days until record expires (default 30)
            
        Returns:
            Saved miner record
        """
        item = {
            'pk': self._make_pk(uid),
            'sk': self._make_sk(block_number, hotkey),
            'uid': uid,
            'hotkey': hotkey,
            'model': model,
            'revision': revision,
            'chute_id': chute_id,
            'chute_slug': chute_slug,
            'model_hash': model_hash,
            'chute_status': chute_status,
            'is_valid': 'true' if is_valid else 'false',  # Store as string for GSI
            'is_valid_bool': is_valid,  # Also store as boolean for convenience
            'invalid_reason': invalid_reason,
            'block_number': block_number,
            'first_block': first_block,
            'ttl': self.get_ttl(ttl_days),
        }
        
        return await self.put(item)
    
    async def get_miner_by_uid(
        self,
        uid: int,
        latest_only: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get miner by UID.
        
        Args:
            uid: Miner UID
            latest_only: If True, return only the latest record (highest block_number)
            
        Returns:
            Latest miner record or None if not found
        """
        pk = self._make_pk(uid)
        
        # Query all records for this UID, sorted by SK (block_number) descending
        records = await self.query(pk=pk, reverse=True, limit=1 if latest_only else None)
        
        if not records:
            return None
        
        return records[0] if latest_only else records
    
    async def get_valid_miners(self) -> List[Dict[str, Any]]:
        """Get all valid miners using GSI.
        
        Returns:
            List of valid miner records
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'is-valid-index',
            'KeyConditionExpression': 'is_valid = :is_valid',
            'ExpressionAttributeValues': {':is_valid': {'S': 'true'}},
            'ScanIndexForward': False  # Sort by block_number descending
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Group by UID and keep only the latest record for each UID
        uid_to_miner = {}
        for item in items:
            uid = item['uid']
            if uid not in uid_to_miner or item['block_number'] > uid_to_miner[uid]['block_number']:
                uid_to_miner[uid] = item
        
        return list(uid_to_miner.values())
    
    async def get_invalid_miners(self) -> List[Dict[str, Any]]:
        """Get all invalid miners using GSI.
        
        Returns:
            List of invalid miner records
        """
        from affine.database.client import get_client
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'is-valid-index',
            'KeyConditionExpression': 'is_valid = :is_valid',
            'ExpressionAttributeValues': {':is_valid': {'S': 'false'}},
            'ScanIndexForward': False
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Group by UID and keep only the latest
        uid_to_miner = {}
        for item in items:
            uid = item['uid']
            if uid not in uid_to_miner or item['block_number'] > uid_to_miner[uid]['block_number']:
                uid_to_miner[uid] = item
        
        return list(uid_to_miner.values())
    
    async def get_miners_by_model_hash(
        self,
        model_hash: str
    ) -> List[Dict[str, Any]]:
        """Get all miners with a specific model hash.
        
        Used for anti-plagiarism detection.
        
        Args:
            model_hash: Model weights SHA256 hash
            
        Returns:
            List of miners with this hash
        """
        from affine.database.client import get_client
        client = get_client()
        
        # Scan table for matching model_hash
        params = {
            'TableName': self.table_name,
            'FilterExpression': 'model_hash = :hash',
            'ExpressionAttributeValues': {':hash': {'S': model_hash}}
        }
        
        response = await client.scan(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        # Group by UID and keep latest
        uid_to_miner = {}
        for item in items:
            uid = item['uid']
            if uid not in uid_to_miner or item['block_number'] > uid_to_miner[uid]['block_number']:
                uid_to_miner[uid] = item
        
        # Sort by first_block (earliest miner first)
        result = sorted(uid_to_miner.values(), key=lambda x: x.get('first_block', float('inf')))
        
        return result
    
    async def delete_old_records(
        self,
        uid: int,
        keep_latest_n: int = 10
    ) -> int:
        """Delete old records for a UID, keeping only the latest N.
        
        Args:
            uid: Miner UID
            keep_latest_n: Number of latest records to keep
            
        Returns:
            Number of records deleted
        """
        pk = self._make_pk(uid)
        
        # Get all records for this UID
        records = await self.query(pk=pk, reverse=True)
        
        if len(records) <= keep_latest_n:
            return 0
        
        # Delete old records
        to_delete = records[keep_latest_n:]
        for record in to_delete:
            await self.delete(record['pk'], record['sk'])
        
        return len(to_delete)