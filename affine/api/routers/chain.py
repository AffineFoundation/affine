"""
Chain Router

Endpoints for querying blockchain information.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from affine.api.dependencies import rate_limit_read
from affine.api.utils.bittensor import get_subtensor

router = APIRouter(prefix="/chain", tags=["Chain"])


class CurrentBlockResponse(BaseModel):
    """Current block number response."""
    
    block_number: int
    timestamp: int


@router.get("/current-block", response_model=CurrentBlockResponse, dependencies=[Depends(rate_limit_read)])
async def get_current_block():
    """
    Get current blockchain block number.
    
    Used by backend services to timestamp samples and scores.
    """
    try:
        import time
        
        # Get subtensor instance
        subtensor = await get_subtensor()
        
        # Query current block
        block_number = await subtensor.get_current_block()
        
        return CurrentBlockResponse(
            block_number=block_number,
            timestamp=int(time.time()),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query current block: {str(e)}"
        )