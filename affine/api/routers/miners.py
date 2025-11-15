"""
Miner Status Router

Endpoints for querying miner status and execution logs.

Note: Miner metadata (uid, stake, etc.) is queried directly from bittensor metagraph,
not stored in database. This router focuses on execution logs and sampling statistics.
"""

import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from affine.api.models import (
    MinerStatusResponse,
    MinerStatistics,
    SamplingSpeed,
    EnvironmentStats,
    RecentError,
)
from affine.api.dependencies import (
    get_execution_logs_dao,
    get_sample_results_dao,
    rate_limit_read,
)
from affine.api.utils.bittensor import query_uid_by_hotkey, query_miner_metadata
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.sample_results import SampleResultsDAO

router = APIRouter(prefix="/miners", tags=["Miners"])


@router.get("/{hotkey}", response_model=MinerStatusResponse, dependencies=[Depends(rate_limit_read)])
async def get_miner_status(
    hotkey: str,
    model_revision: str = Query(..., description="Model revision (required)"),
    logs_dao: ExecutionLogsDAO = Depends(get_execution_logs_dao),
    samples_dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Get miner status and execution statistics.
    
    Returns comprehensive information about a miner including:
    - UID (from bittensor metagraph)
    - Execution statistics (success/failure counts)
    - Sampling speed
    - Error history
    
    Query parameters:
    - model_revision: Model revision to query (required)
    
    Note: Miner metadata (uid, stake, etc.) is queried from bittensor metagraph in real-time.
    """
    try:
        # Query bittensor metagraph for miner metadata
        uid = await query_uid_by_hotkey(hotkey)
        bt_metadata = await query_miner_metadata(uid) if uid is not None else None
        
        if not bt_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner {hotkey} not found in bittensor metagraph"
            )
        
        # Get execution statistics (last hour)
        stats = await logs_dao.get_execution_stats(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            time_window_seconds=3600,
        )
        
        # Get recent errors
        recent_errors = await logs_dao.get_recent_logs(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            limit=10,
            status='failed',
        )
        
        # Calculate sampling speed from actual sample counts
        # Query sample counts for different time windows
        now = int(time.time() * 1000)
        hour_ago = now - (3600 * 1000)
        day_ago = now - (24 * 3600 * 1000)
        week_ago = now - (7 * 24 * 3600 * 1000)
        
        # TODO: Implement count queries in SampleResultsDAO
        # For now, use placeholder values
        sampling_speed = SamplingSpeed(
            last_hour=0.0,
            last_day=0.0,
            last_week=0.0,
        )
        
        # Parse environment stats
        env_stats = {}
        for env, counts in stats.get("by_environment", {}).items():
            env_stats[env] = EnvironmentStats(
                success=counts.get("success", 0),
                failure=counts.get("failure", 0),
            )
        
        # Parse recent errors
        error_list = [
            RecentError(
                timestamp=err["timestamp"],
                error_type=err.get("error_type", "UNKNOWN"),
                error_message=err.get("error_message", ""),
            )
            for err in recent_errors
        ]
        
        # Build statistics
        statistics = MinerStatistics(
            total_samples=stats.get("total_samples", 0),
            success_count=stats.get("success_count", 0),
            error_count=stats.get("error_count", 0),
            consecutive_errors=stats.get("consecutive_errors", 0),
            sampling_speed=sampling_speed,
            by_environment=env_stats,
            recent_errors=error_list,
        )
        
        return MinerStatusResponse(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            model="unknown",  # Could be stored in system_config if needed
            chutes_status="unknown",  # Could query chutes API if needed
            is_paused=False,  # Pause status removed (use blacklist in system_config instead)
            pause_until=None,
            statistics=statistics,
            last_updated=int(time.time()),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve miner status: {str(e)}"
        )