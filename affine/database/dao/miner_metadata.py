"""
Miner Metadata DAO

Manages miner metadata including chutes status and statistics.
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class MinerMetadataDAO(BaseDAO):
    """DAO for miner_metadata table.
    
    Stores miner metadata and operational status.
    PK: MINER#{hotkey}
    SK: METADATA
    """
    
    def __init__(self):
        self.table_name = get_table_name("miner_metadata")
        super().__init__()
    
    def _make_pk(self, miner_hotkey: str) -> str:
        """Generate partition key."""
        return f"MINER#{miner_hotkey}"
    
    def _make_sk(self) -> str:
        """Generate sort key."""
        return "METADATA"
    
    async def save_metadata(
        self,
        miner_hotkey: str,
        uid: int,
        current_revision: str,
        model: str,
        model_name: str,
        chutes_slug: Optional[str] = None,
        chutes_status: str = "unknown",
        last_commit_block: Optional[int] = None,
        first_commit_block: Optional[int] = None
    ) -> Dict[str, Any]:
        """Save or update miner metadata.
        
        Args:
            miner_hotkey: Miner's hotkey
            uid: Miner UID
            current_revision: Current model revision
            model: Current model repo/name
            model_name: Model name
            chutes_slug: Chutes slug if available
            chutes_status: Chutes status (hot/cold/unknown)
            last_commit_block: Last commit block number
            first_commit_block: First commit block number
            
        Returns:
            Saved metadata item
        """
        # Get existing metadata to preserve statistics
        existing = await self.get_metadata(miner_hotkey)
        
        updated_at = int(time.time())
        
        item = {
            'pk': self._make_pk(miner_hotkey),
            'sk': self._make_sk(),
            'miner_hotkey': miner_hotkey,
            'uid': uid,
            'current_revision': current_revision,
            'model': model,
            'model_name': model_name,
            'chutes_slug': chutes_slug,
            'chutes_status': chutes_status,
            'chutes_last_checked': updated_at,
            'last_commit_block': last_commit_block,
            'first_commit_block': first_commit_block or last_commit_block,
            'updated_at': updated_at,
        }
        
        # Preserve statistics if they exist
        if existing:
            item['total_samples'] = existing.get('total_samples', 0)
            item['success_count'] = existing.get('success_count', 0)
            item['error_count'] = existing.get('error_count', 0)
            item['last_sample_time'] = existing.get('last_sample_time')
            item['is_paused'] = existing.get('is_paused', False)
            item['pause_reason'] = existing.get('pause_reason')
            item['pause_until'] = existing.get('pause_until')
            item['consecutive_errors'] = existing.get('consecutive_errors', 0)
        else:
            item['total_samples'] = 0
            item['success_count'] = 0
            item['error_count'] = 0
            item['last_sample_time'] = None
            item['is_paused'] = False
            item['pause_reason'] = None
            item['pause_until'] = None
            item['consecutive_errors'] = 0
        
        return await self.put(item)
    
    async def get_metadata(self, miner_hotkey: str) -> Optional[Dict[str, Any]]:
        """Get miner metadata.
        
        Args:
            miner_hotkey: Miner's hotkey
            
        Returns:
            Metadata if found, None otherwise
        """
        pk = self._make_pk(miner_hotkey)
        sk = self._make_sk()
        
        return await self.get(pk, sk)
    
    async def update_statistics(
        self,
        miner_hotkey: str,
        success: bool,
        increment_samples: bool = True
    ) -> bool:
        """Update miner statistics after sampling.
        
        Args:
            miner_hotkey: Miner's hotkey
            success: Whether sampling succeeded
            increment_samples: Whether to increment total sample count
            
        Returns:
            True if updated successfully
        """
        metadata = await self.get_metadata(miner_hotkey)
        if not metadata:
            return False
        
        if increment_samples:
            metadata['total_samples'] = metadata.get('total_samples', 0) + 1
        
        if success:
            metadata['success_count'] = metadata.get('success_count', 0) + 1
            metadata['consecutive_errors'] = 0
        else:
            metadata['error_count'] = metadata.get('error_count', 0) + 1
            metadata['consecutive_errors'] = metadata.get('consecutive_errors', 0) + 1
        
        metadata['last_sample_time'] = int(time.time())
        metadata['updated_at'] = int(time.time())
        
        await self.put(metadata)
        return True
    
    async def pause_miner(
        self,
        miner_hotkey: str,
        reason: str,
        duration_seconds: Optional[int] = None
    ) -> bool:
        """Pause a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            reason: Reason for pause
            duration_seconds: Optional pause duration (None = indefinite)
            
        Returns:
            True if paused successfully
        """
        metadata = await self.get_metadata(miner_hotkey)
        if not metadata:
            return False
        
        metadata['is_paused'] = True
        metadata['pause_reason'] = reason
        
        if duration_seconds:
            metadata['pause_until'] = int(time.time()) + duration_seconds
        else:
            metadata['pause_until'] = None
        
        metadata['updated_at'] = int(time.time())
        
        await self.put(metadata)
        return True
    
    async def unpause_miner(self, miner_hotkey: str) -> bool:
        """Unpause a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            
        Returns:
            True if unpaused successfully
        """
        metadata = await self.get_metadata(miner_hotkey)
        if not metadata:
            return False
        
        metadata['is_paused'] = False
        metadata['pause_reason'] = None
        metadata['pause_until'] = None
        metadata['consecutive_errors'] = 0
        metadata['updated_at'] = int(time.time())
        
        await self.put(metadata)
        return True
    
    async def check_pause_expired(self, miner_hotkey: str) -> bool:
        """Check if pause duration has expired and auto-unpause.
        
        Args:
            miner_hotkey: Miner's hotkey
            
        Returns:
            True if miner was auto-unpaused
        """
        metadata = await self.get_metadata(miner_hotkey)
        if not metadata or not metadata.get('is_paused'):
            return False
        
        pause_until = metadata.get('pause_until')
        if pause_until and int(time.time()) >= pause_until:
            await self.unpause_miner(miner_hotkey)
            return True
        
        return False
    
    async def get_all_miners(self) -> List[Dict[str, Any]]:
        """Get all miner metadata.
        
        Returns:
            List of all miner metadata
        """
        # This requires scanning the table
        # For better performance in production, consider maintaining a list in system_config
        client = self.get_client()
        
        params = {
            'TableName': self.table_name,
            'FilterExpression': 'sk = :sk',
            'ExpressionAttributeValues': {
                ':sk': {'S': 'METADATA'}
            }
        }
        
        items = []
        
        while True:
            response = await client.scan(**params)
            items.extend([self._deserialize(item) for item in response.get('Items', [])])
            
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break
            
            params['ExclusiveStartKey'] = last_key
        
        return items