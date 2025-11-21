"""
Scores Router

Endpoints for querying score calculations.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from affine.api.models import (
    ScoresResponse,
    MinerScoreHistoryResponse,
    MinerScore,
    MinerScoreHistory,
)
from affine.api.dependencies import (
    get_scores_dao,
    get_score_snapshots_dao,
    rate_limit_read,
)
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO

router = APIRouter(prefix="/scores", tags=["Scores"])


@router.get("/latest", response_model=ScoresResponse, dependencies=[Depends(rate_limit_read)])
async def get_latest_scores(
    limit: int = Query(256, description="Maximum miners to return", ge=1, le=512),
    dao: ScoresDAO = Depends(get_scores_dao),
):
    """
    Get the most recent score snapshot.
    
    Returns scores for all miners at the latest calculated block.
    
    Query parameters:
    - limit: Maximum number of miners to return (default: 256, max: 512)
    """
    try:
        scores_data = await dao.get_latest_scores(limit=limit)
        
        if not scores_data or not scores_data.get('block_number'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No scores found"
            )
        
        # Parse scores data
        block_number = scores_data.get("block_number")
        calculated_at = scores_data.get("calculated_at")
        scores_list = scores_data.get("scores", [])
        
        # Limit results
        scores_list = scores_list[:limit]
        
        # Convert to response models with safe field access
        miner_scores = [
            MinerScore(
                miner_hotkey=s.get("miner_hotkey", ""),
                uid=s.get("uid", 0),
                model_revision=s.get("model_revision", "unknown"),
                overall_score=s.get("overall_score", 0.0),
                average_score=s.get("average_score", 0.0),
                confidence_interval=s.get("confidence_interval", [0.0, 0.0]),
                scores_by_layer=s.get("scores_by_layer", {}),
                scores_by_env=s.get("scores_by_env", {}),
                total_samples=s.get("total_samples", 0),
                is_eligible=s.get("is_eligible", False),
                meets_criteria=s.get("meets_criteria", False),
            )
            for s in scores_list
        ]
        
        return ScoresResponse(
            block_number=block_number,
            calculated_at=calculated_at,
            scores=miner_scores,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest scores: {str(e)}"
        )


@router.get("/block/{block_number}", response_model=ScoresResponse, dependencies=[Depends(rate_limit_read)])
async def get_scores_at_block(
    block_number: int,
    dao: ScoresDAO = Depends(get_scores_dao),
):
    """
    Get scores at a specific block.
    
    Returns the score snapshot calculated for the specified block number.
    """
    try:
        scores_list = await dao.get_scores_at_block(block_number)
        
        if not scores_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No scores found for block {block_number}"
            )
        
        # Get calculated_at from first score
        calculated_at = scores_list[0].get("calculated_at") if scores_list else None
        
        # Convert to response models with safe field access
        miner_scores = [
            MinerScore(
                miner_hotkey=s.get("miner_hotkey", ""),
                uid=s.get("uid", 0),
                model_revision=s.get("model_revision", "unknown"),
                overall_score=s.get("overall_score", 0.0),
                average_score=s.get("average_score", 0.0),
                confidence_interval=s.get("confidence_interval", [0.0, 0.0]),
                scores_by_layer=s.get("scores_by_layer", {}),
                scores_by_env=s.get("scores_by_env", {}),
                total_samples=s.get("total_samples", 0),
                is_eligible=s.get("is_eligible", False),
                meets_criteria=s.get("meets_criteria", False),
            )
            for s in scores_list
        ]
        
        return ScoresResponse(
            block_number=block_number,
            calculated_at=calculated_at,
            scores=miner_scores,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve scores for block {block_number}: {str(e)}"
        )


@router.get("/miner/{hotkey}", response_model=MinerScoreHistoryResponse, dependencies=[Depends(rate_limit_read)])
async def get_miner_score_history(
    hotkey: str,
    num_blocks: int = Query(10, description="Number of recent blocks", ge=1, le=100),
    dao: ScoresDAO = Depends(get_scores_dao),
):
    """
    Get score history for a miner.
    
    Returns the miner's scores across recent blocks.
    
    Query parameters:
    - num_blocks: Number of recent blocks to retrieve (default: 10, max: 100)
    """
    try:
        history_data = await dao.get_miner_score_history(hotkey, num_blocks)
        
        # Convert to response models with safe field access
        history = [
            MinerScoreHistory(
                block_number=h.get("block_number", 0),
                calculated_at=h.get("calculated_at", 0),
                overall_score=h.get("overall_score", 0.0),
                average_score=h.get("average_score", 0.0),
                confidence_interval=h.get("confidence_interval", [0.0, 0.0]),
            )
            for h in history_data
        ]
        
        return MinerScoreHistoryResponse(
            miner_hotkey=hotkey,
            history=history,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve score history: {str(e)}"
        )


@router.get("/weights/latest", dependencies=[Depends(rate_limit_read)])
async def get_latest_weights(
    snapshots_dao: ScoreSnapshotsDAO = Depends(get_score_snapshots_dao),
):
    """
    Get the latest normalized weights from scoring calculation.
    
    Returns the most recent score snapshot with normalized weights
    for all miners, suitable for setting on-chain weights.
    
    Response format:
    {
        "block_number": 12345,
        "weights": {
            "0": {"hotkey": "5...", "weight": 0.15},
            "1": {"hotkey": "5...", "weight": 0.12},
            ...
        }
    }
    """
    try:
        # Get latest snapshot
        snapshot = await snapshots_dao.get_latest_snapshot()
        
        if not snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No score snapshots found"
            )
        
        # Extract weights from statistics
        statistics = snapshot.get('statistics', {})
        miner_weights = statistics.get('miner_final_scores', {})
        
        if not miner_weights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No weights found in latest snapshot"
            )
        
        # Format response according to design document
        weights_response = {}
        for uid_str, weight in miner_weights.items():
            # Get hotkey from snapshot metadata if available
            # For now, use uid as key
            weights_response[uid_str] = {
                "weight": weight
            }
        
        return {
            "block_number": snapshot.get('block_number'),
            "weights": weights_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve latest weights: {str(e)}"
        )