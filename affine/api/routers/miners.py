"""
Miner Status Router

Endpoints for querying miner status and execution logs.

Note: Miner metadata (uid, stake, etc.) is queried directly from bittensor metagraph,
not stored in database. This router focuses on execution logs and sampling statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from affine.api.dependencies import (
    rate_limit_read,
)
from affine.api.models import MinerResponse

router = APIRouter(prefix="/miners", tags=["Miners"])

@router.get("/uid/{uid}", response_model=MinerResponse, dependencies=[Depends(rate_limit_read)])
async def get_miner_by_uid(
    uid: int,
):
    """
    Get miner information by UID from MinersMonitor.
    
    Returns complete miner info including:
    - hotkey: Miner's hotkey
    - uid: Miner's UID
    - model: Model name (HuggingFace repo)
    - revision: Model revision hash
    - chute_id: Chute deployment ID
    - block_number: Block number when discovered
    - first_block: First block number when discovered
    - is_valid: Validation status
    - invalid_reason: Reason for validation failure (if any)
    - model_hash: Hash of model weights for plagiarism detection
    - chute_slug: Chute slug identifier
    - chute_status: Chute deployment status
    """
    try:
        # Query miner by UID from database
        from affine.database.dao.miners import MinersDAO
        miners_dao = MinersDAO()
        
        miner = await miners_dao.get_miner_by_uid(uid)
        
        if not miner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner with UID {uid} not found"
            )

        # Build response using MinerResponse model (automatically excludes 'pk')
        return MinerResponse(**miner)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve miner info: {str(e)}"
        )