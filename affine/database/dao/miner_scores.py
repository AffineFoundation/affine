"""
Miner Scores DAO

Manages detailed scoring records for each miner at each block.
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client


class MinerScoresDAO(BaseDAO):
    """DAO for miner_scores table.
    
    Stores detailed scoring information for miners at each block snapshot.
    
    PK: SNAPSHOT#{block_number}
    SK: MINER#{hotkey}
    """
    
    def __init__(self):
        self.table_name = get_table_name("miner_scores")
        super().__init__()
    
    def _make_pk(self, block_number: int) -> str:
        """Generate partition key."""
        return f"SNAPSHOT#{block_number}"
    
    def _make_sk(self, hotkey: str) -> str:
        """Generate sort key."""
        return f"MINER#{hotkey}"
    
    async def save_miner_score(
        self,
        block_number: int,
        hotkey: str,
        uid: int,
        model_revision: str,
        env_scores: Dict[str, Dict[str, Any]],
        layer_scores: Dict[str, float],
        subset_contributions: Dict[str, Dict[str, Any]],
        cumulative_weight: float,
        normalized_weight: float,
        filter_info: Optional[Dict[str, Any]] = None,
        ttl_days: int = 30
    ) -> Dict[str, Any]:
        """Save a miner's score record for a specific block.
        
        Args:
            block_number: Block number for this snapshot
            hotkey: Miner's hotkey
            uid: Miner's UID
            model_revision: Model revision
            env_scores: Scores by environment
                {
                    "env_name": {
                        "average_score": float,
                        "sample_count": int,
                        "completeness": float
                    }
                }
            layer_scores: Scores by layer (L1, L2, L3, ...)
            subset_contributions: Contributions by subset
                {
                    "subset_key": {
                        "geometric_mean": float,
                        "rank": int,
                        "weight": float
                    }
                }
            cumulative_weight: Total accumulated weight
            normalized_weight: Final normalized weight
            filter_info: Information about filtering
                {
                    "filtered_subsets": [subset_key, ...],
                    "filter_reasons": {subset_key: reason}
                }
            ttl_days: Days until automatic deletion (default 30)
            
        Returns:
            Saved item
        """
        calculated_at = int(time.time())
        
        item = {
            'pk': self._make_pk(block_number),
            'sk': self._make_sk(hotkey),
            'block_number': block_number,
            'hotkey': hotkey,
            'uid': uid,
            'model_revision': model_revision,
            'calculated_at': calculated_at,
            'env_scores': env_scores,
            'layer_scores': layer_scores,
            'subset_contributions': subset_contributions,
            'cumulative_weight': cumulative_weight,
            'normalized_weight': normalized_weight,
            'ttl': self.get_ttl(ttl_days),
        }
        
        if filter_info:
            item['filter_info'] = filter_info
        
        return await self.put(item)
    
    async def get_scores_at_block(
        self,
        block_number: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all miner scores at a specific block.
        
        Args:
            block_number: Block number
            limit: Maximum number of results
            
        Returns:
            List of miner score records
        """
        pk = self._make_pk(block_number)
        return await self.query(pk=pk, limit=limit)
    
    async def get_miner_score_at_block(
        self,
        block_number: int,
        hotkey: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific miner's score at a block.
        
        Args:
            block_number: Block number
            hotkey: Miner's hotkey
            
        Returns:
            Miner score record if found, None otherwise
        """
        pk = self._make_pk(block_number)
        sk = self._make_sk(hotkey)
        return await self.get(pk, sk)
    
    async def get_miner_history(
        self,
        hotkey: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get score history for a miner.
        
        Uses hotkey-index GSI to query miner's historical records.
        
        Args:
            hotkey: Miner's hotkey
            limit: Maximum number of records
            
        Returns:
            List of historical score records (sorted by block_number desc)
        """
        client = get_client()
        
        params = {
            'TableName': self.table_name,
            'IndexName': 'hotkey-index',
            'KeyConditionExpression': 'hotkey = :hotkey',
            'ExpressionAttributeValues': {
                ':hotkey': {'S': hotkey}
            },
            'ScanIndexForward': False,  # Descending order (newest first)
            'Limit': limit
        }
        
        response = await client.query(**params)
        items = [self._deserialize(item) for item in response.get('Items', [])]
        
        return items
    
    async def get_latest_block_number(self) -> Optional[int]:
        """Get the latest block number with score records.
        
        Returns:
            Latest block number, or None if no records exist
        """
        client = get_client()
        
        # Query one record from block-number-index in descending order
        params = {
            'TableName': self.table_name,
            'IndexName': 'block-number-index',
            'Limit': 1,
            'ScanIndexForward': False,  # Descending
        }
        
        response = await client.scan(**params)
        items = response.get('Items', [])
        
        if not items:
            return None
        
        item = self._deserialize(items[0])
        return item.get('block_number')
    
    async def delete_scores_at_block(self, block_number: int) -> int:
        """Delete all scores at a specific block.
        
        Args:
            block_number: Block number
            
        Returns:
            Number of records deleted
        """
        scores = await self.get_scores_at_block(block_number)
        
        deleted_count = 0
        for score in scores:
            await self.delete(score['pk'], score['sk'])
            deleted_count += 1
        
        return deleted_count