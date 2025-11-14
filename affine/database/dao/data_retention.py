"""
Data Retention Policy DAO

Manages data retention policies for miners.
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class DataRetentionDAO(BaseDAO):
    """DAO for data_retention_policy table.
    
    Manages retention policies to protect historical top-3 miners.
    PK: RETENTION#{hotkey}
    SK: METADATA
    """
    
    def __init__(self):
        self.table_name = get_table_name("data_retention_policy")
        super().__init__()
    
    def _make_pk(self, miner_hotkey: str) -> str:
        """Generate partition key."""
        return f"RETENTION#{miner_hotkey}"
    
    def _make_sk(self) -> str:
        """Generate sort key."""
        return "METADATA"
    
    async def set_policy(
        self,
        miner_hotkey: str,
        is_protected: bool,
        protection_reason: Optional[str] = None,
        retention_days: int = 90
    ) -> Dict[str, Any]:
        """Set retention policy for a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            is_protected: Whether data should be protected from cleanup
            protection_reason: Reason for protection (e.g., "Historical rank #1")
            retention_days: Days to retain data if not protected
            
        Returns:
            Saved policy item
        """
        # Get existing policy to preserve protected_since
        existing = await self.get_policy(miner_hotkey)
        
        item = {
            'pk': self._make_pk(miner_hotkey),
            'sk': self._make_sk(),
            'miner_hotkey': miner_hotkey,
            'is_protected': is_protected,
            'protection_reason': protection_reason,
            'retention_days': retention_days,
            'last_cleanup_at': existing.get('last_cleanup_at') if existing else None,
        }
        
        # Preserve protected_since if already protected, otherwise set it
        if is_protected:
            if existing and existing.get('is_protected'):
                item['protected_since'] = existing.get('protected_since')
            else:
                item['protected_since'] = int(time.time())
        else:
            item['protected_since'] = None
        
        return await self.put(item)
    
    async def get_policy(self, miner_hotkey: str) -> Optional[Dict[str, Any]]:
        """Get retention policy for a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            
        Returns:
            Policy if found, None otherwise (defaults will apply)
        """
        pk = self._make_pk(miner_hotkey)
        sk = self._make_sk()
        
        return await self.get(pk, sk)
    
    async def is_protected(self, miner_hotkey: str) -> bool:
        """Check if a miner's data is protected.
        
        Args:
            miner_hotkey: Miner's hotkey
            
        Returns:
            True if protected, False otherwise
        """
        policy = await self.get_policy(miner_hotkey)
        return policy.get('is_protected', False) if policy else False
    
    async def set_protected(
        self,
        miner_hotkey: str,
        reason: str,
        protected_since: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mark a miner as protected (e.g., historical top-3).
        
        Args:
            miner_hotkey: Miner's hotkey
            reason: Reason for protection
            protected_since: Optional timestamp (defaults to now)
            
        Returns:
            Updated policy
        """
        if protected_since is None:
            protected_since = int(time.time())
        
        return await self.set_policy(
            miner_hotkey=miner_hotkey,
            is_protected=True,
            protection_reason=reason
        )
    
    async def set_unprotected(self, miner_hotkey: str) -> Dict[str, Any]:
        """Remove protection from a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            
        Returns:
            Updated policy
        """
        return await self.set_policy(
            miner_hotkey=miner_hotkey,
            is_protected=False,
            protection_reason=None
        )
    
    async def update_cleanup_timestamp(self, miner_hotkey: str) -> bool:
        """Update last cleanup timestamp for a miner.
        
        Args:
            miner_hotkey: Miner's hotkey
            
        Returns:
            True if updated successfully
        """
        policy = await self.get_policy(miner_hotkey)
        if not policy:
            # Create default policy
            policy = await self.set_policy(
                miner_hotkey=miner_hotkey,
                is_protected=False
            )
        
        policy['last_cleanup_at'] = int(time.time())
        
        await self.put(policy)
        return True
    
    async def get_all_policies(self) -> List[Dict[str, Any]]:
        """Get all retention policies.
        
        Returns:
            List of all policies
        """
        # Scan table for all policies
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
    
    async def get_protected_miners(self) -> List[str]:
        """Get list of protected miner hotkeys.
        
        Returns:
            List of protected miner hotkeys
        """
        policies = await self.get_all_policies()
        
        return [
            p['miner_hotkey']
            for p in policies
            if p.get('is_protected', False)
        ]