"""
Scores DAO

Manages score snapshots organized by block number.
"""

import time
from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name
from affine.database.client import get_client


class ScoresDAO(BaseDAO):
    """DAO for scores table.
    
    Stores score snapshots per block with 30-day TTL.
    PK: SCORE#{block_number}
    SK: MINER#{hotkey}
    """
    
    def __init__(self):
        self.table_name = get_table_name("scores")
        super().__init__()
    
    def _make_pk(self, block_number: int) -> str:
        """Generate partition key."""
        return f"SCORE#{block_number}"
    
    def _make_sk(self, miner_hotkey: str) -> str:
        """Generate sort key."""
        return f"MINER#{miner_hotkey}"
    
    async def save_score(
        self,
        block_number: int,
        miner_hotkey: str,
        uid: int,
        model_revision: str,
        overall_score: float,
        confidence_interval_lower: float,
        confidence_interval_upper: float,
        average_score: float,
        scores_by_layer: Dict[str, float],
        scores_by_env: Dict[str, float],
        total_samples: int,
        is_eligible: bool,
        meets_criteria: bool
    ) -> Dict[str, Any]:
        """Save a score snapshot for a miner at a specific block.
        
        Args:
            block_number: Current block number
            miner_hotkey: Miner's hotkey
            uid: Miner UID
            model_revision: Model revision
            overall_score: Overall score
            confidence_interval_lower: Lower bound of CI
            confidence_interval_upper: Upper bound of CI
            average_score: Average score
            scores_by_layer: Scores breakdown by layer
            scores_by_env: Scores breakdown by environment
            total_samples: Total number of samples
            is_eligible: Whether miner is eligible for rewards
            meets_criteria: Whether miner meets all criteria
            
        Returns:
            Saved score item
        """
        calculated_at = int(time.time())
        
        item = {
            'pk': self._make_pk(block_number),
            'sk': self._make_sk(miner_hotkey),
            'block_number': block_number,
            'miner_hotkey': miner_hotkey,
            'uid': uid,
            'model_revision': model_revision,
            'calculated_at': calculated_at,
            'overall_score': overall_score,
            'confidence_interval_lower': confidence_interval_lower,
            'confidence_interval_upper': confidence_interval_upper,
            'average_score': average_score,
            'scores_by_layer': scores_by_layer,
            'scores_by_env': scores_by_env,
            'total_samples': total_samples,
            'is_eligible': is_eligible,
            'meets_criteria': meets_criteria,
            'latest_marker': 'LATEST',  # For GSI
            'ttl': self.get_ttl(30),  # 30 days
        }
        
        return await self.put(item)
    
    async def get_scores_at_block(
        self,
        block_number: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all miner scores at a specific block.
        
        Args:
            block_number: Block number
            limit: Maximum number of scores to return
            
        Returns:
            List of score entries
        """
        pk = self._make_pk(block_number)
        
        return await self.query(pk=pk, limit=limit)
    
    async def get_latest_scores(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get the most recent score snapshot.
        
        Uses latest-block-index GSI to find the most recent block.
        
        Args:
            limit: Maximum number of scores to return
            
        Returns:
            Dictionary with block_number and scores list
        """
        client = get_client()
        
        # Query GSI to get latest block
        params = {
            'TableName': self.table_name,
            'IndexName': 'latest-block-index',
            'KeyConditionExpression': 'latest_marker = :marker',
            'ExpressionAttributeValues': {
                ':marker': {'S': 'LATEST'}
            },
            'ScanIndexForward': False,  # Descending order
            'Limit': 1
        }
        
        response = await client.query(**params)
        items = response.get('Items', [])
        
        if not items:
            return {'block_number': None, 'scores': []}
        
        # Get the latest block number
        latest_item = self._deserialize(items[0])
        latest_block = latest_item['block_number']
        
        # Get all scores for this block
        scores = await self.get_scores_at_block(latest_block, limit=limit)
        
        return {
            'block_number': latest_block,
            'calculated_at': latest_item['calculated_at'],
            'scores': scores
        }
    
    async def get_miner_score_at_block(
        self,
        block_number: int,
        miner_hotkey: str
    ) -> Optional[Dict[str, Any]]:
        """Get a specific miner's score at a block.
        
        Args:
            block_number: Block number
            miner_hotkey: Miner's hotkey
            
        Returns:
            Score entry if found, None otherwise
        """
        pk = self._make_pk(block_number)
        sk = self._make_sk(miner_hotkey)
        
        return await self.get(pk, sk)
    
    async def get_miner_score_history(
        self,
        miner_hotkey: str,
        num_blocks: int = 10
    ) -> List[Dict[str, Any]]:
        """Get score history for a miner across recent blocks.
        
        Note: This requires scanning multiple partitions.
        For efficient queries, use get_latest_scores() instead.
        
        Args:
            miner_hotkey: Miner's hotkey
            num_blocks: Number of recent blocks to fetch
            
        Returns:
            List of score entries
        """
        # Get latest block first
        latest = await self.get_latest_scores(limit=1)
        if not latest['block_number']:
            return []
        
        latest_block = latest['block_number']
        
        # Fetch scores from recent blocks
        history = []
        for i in range(num_blocks):
            block = latest_block - (i * 200)  # Approximate 30-minute intervals
            score = await self.get_miner_score_at_block(block, miner_hotkey)
            if score:
                history.append(score)
        
        return history
    
    async def save_weight_snapshot(
        self,
        block_number: int,
        weights: Dict[str, float],
        calculation_details: Dict[str, Any],
        scorer_hotkey: str = "scorer_service"
    ) -> Dict[str, Any]:
        """Save a complete weight snapshot for all miners.
        
        This is a convenience method that saves weights for all miners at once.
        
        Args:
            block_number: Current block number
            weights: Dict mapping hotkey -> weight (0.0 to 1.0)
            calculation_details: Details about the calculation (method, params, etc.)
            scorer_hotkey: Service identifier
            
        Returns:
            Summary of saved snapshot
        """
        import uuid
        snapshot_id = str(uuid.uuid4())
        created_at = int(time.time())
        
        # Save each miner's weight
        saved_count = 0
        for hotkey, weight in weights.items():
            # Get additional info from calculation_details if available
            miner_details = calculation_details.get('miners', {}).get(hotkey, {})
            
            item = {
                'pk': self._make_pk(block_number),
                'sk': self._make_sk(hotkey),
                'block_number': block_number,
                'miner_hotkey': hotkey,
                'uid': miner_details.get('uid', -1),
                'model_revision': miner_details.get('model_revision', ''),
                'calculated_at': created_at,
                'overall_score': weight,
                'confidence_interval_lower': miner_details.get('ci_lower', weight),
                'confidence_interval_upper': miner_details.get('ci_upper', weight),
                'average_score': miner_details.get('average_score', weight),
                'scores_by_layer': miner_details.get('scores_by_layer', {}),
                'scores_by_env': miner_details.get('scores_by_env', {}),
                'total_samples': miner_details.get('total_samples', 0),
                'is_eligible': miner_details.get('is_eligible', True),
                'meets_criteria': miner_details.get('meets_criteria', True),
                'latest_marker': 'LATEST',
                'ttl': self.get_ttl(30),
                'snapshot_id': snapshot_id,
            }
            
            await self.put(item)
            saved_count += 1
        
        return {
            'snapshot_id': snapshot_id,
            'block_number': block_number,
            'created_at': created_at,
            'miners_count': saved_count,
            'calculation_details': calculation_details
        }
    
    async def get_weights_for_setting(self) -> Dict[str, Any]:
        """Get the latest weights in a format suitable for chain setting.
        
        Returns:
            Dict with:
                - block_number: Block at which weights were calculated
                - weights: Dict mapping hotkey -> weight
                - uids: Dict mapping uid -> weight (for chain setting)
        """
        latest = await self.get_latest_scores()
        
        if not latest['block_number']:
            return {
                'block_number': None,
                'weights': {},
                'uids': {}
            }
        
        weights_by_hotkey = {}
        weights_by_uid = {}
        
        for score in latest['scores']:
            hotkey = score.get('miner_hotkey')
            uid = score.get('uid', -1)
            weight = score.get('overall_score', 0.0)
            
            if hotkey:
                weights_by_hotkey[hotkey] = weight
            
            if uid >= 0:
                weights_by_uid[uid] = weight
        
        return {
            'block_number': latest['block_number'],
            'calculated_at': latest.get('calculated_at'),
            'weights': weights_by_hotkey,
            'uids': weights_by_uid
        }