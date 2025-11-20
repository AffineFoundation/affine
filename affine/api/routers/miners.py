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
from typing import Dict, Any

router = APIRouter(prefix="/miners", tags=["Miners"])

@router.get("/uid/{uid}", response_model=Dict[str, Any], dependencies=[Depends(rate_limit_read)])
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
    - is_valid: Validation status
    - invalid_reason: Reason for validation failure (if any)
    - model_hash: Hash of model weights for plagiarism detection
    - discovered_at: Timestamp when first discovered
    - last_updated: Timestamp of last update
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
        
        # Return miner info
        return {
            "hotkey": miner.get("hotkey"),
            "uid": miner.get("uid"),
            "model": miner.get("model"),
            "revision": miner.get("revision"),
            "chute_id": miner.get("chute_id"),
            "block_number": miner.get("block_number"),
            "is_valid": miner.get("is_valid") == "true",  # Convert string to bool
            "invalid_reason": miner.get("invalid_reason"),
            "model_hash": miner.get("model_hash"),
            "discovered_at": miner.get("discovered_at"),
            "last_updated": miner.get("last_updated"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve miner info: {str(e)}"
        )