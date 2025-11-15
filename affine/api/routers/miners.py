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
    MinerIsPausedResponse,
    ConsecutiveErrorsResponse,
    MinerPauseRequest,
    MinerPauseResponse,
    MinerStatsResponse,
)
from affine.api.dependencies import (
    get_execution_logs_dao,
    get_sample_results_dao,
    get_system_config_dao,
    rate_limit_read,
)
from affine.api.utils.bittensor import query_uid_by_hotkey, query_miner_metadata
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.system_config import SystemConfigDAO

router = APIRouter(prefix="/miners", tags=["Miners"])

# Dataset lengths (should be loaded from config)
DATASET_LENGTHS = {
    "affine:sat": 200,
    "affine:abd": 200,
    "affine:ded": 200,
}


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


@router.get("/{hotkey}/is-paused", response_model=MinerIsPausedResponse, dependencies=[Depends(rate_limit_read)])
async def check_miner_is_paused(
    hotkey: str,
    config_dao: SystemConfigDAO = Depends(get_system_config_dao),
):
    """
    Check if a miner is currently paused.
    
    Returns pause status and expiration time if paused.
    """
    try:
        # Get pause status from system_config
        pause_key = f"miner_pause:{hotkey}"
        pause_data = await config_dao.get_param_value(pause_key)
        
        if not pause_data:
            return MinerIsPausedResponse(
                miner_hotkey=hotkey,
                is_paused=False,
                paused_until=None,
                reason=None,
            )
        
        # Check if pause has expired
        paused_until = pause_data.get("paused_until", 0)
        current_time = int(time.time())
        
        if paused_until <= current_time:
            # Pause has expired, clean up
            await config_dao.delete_param(pause_key)
            return MinerIsPausedResponse(
                miner_hotkey=hotkey,
                is_paused=False,
                paused_until=None,
                reason=None,
            )
        
        return MinerIsPausedResponse(
            miner_hotkey=hotkey,
            is_paused=True,
            paused_until=paused_until,
            reason=pause_data.get("reason"),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check pause status: {str(e)}"
        )


@router.get("/{hotkey}/consecutive-errors", response_model=ConsecutiveErrorsResponse, dependencies=[Depends(rate_limit_read)])
async def get_consecutive_errors(
    hotkey: str,
    logs_dao: ExecutionLogsDAO = Depends(get_execution_logs_dao),
):
    """
    Get consecutive error count for a miner.
    
    Used by executor to determine if miner should be paused.
    """
    try:
        # Get recent logs (limit 20 to check for consecutive errors)
        logs = await logs_dao.get_recent_logs(hotkey, limit=20)
        
        # Count consecutive errors
        consecutive_count = 0
        recent_errors = []
        
        for log in logs:
            if log.get("status") == "failed":
                consecutive_count += 1
                if len(recent_errors) < 10:  # Keep last 10 errors
                    recent_errors.append(RecentError(
                        timestamp=log["timestamp"],
                        error_type=log.get("error_type", "UNKNOWN"),
                        error_message=log.get("error_message", ""),
                    ))
            else:
                break  # Stop counting on first success
        
        # Default threshold (should be loaded from config)
        threshold = 3
        
        return ConsecutiveErrorsResponse(
            miner_hotkey=hotkey,
            consecutive_errors=consecutive_count,
            threshold=threshold,
            should_pause=consecutive_count >= threshold,
            recent_errors=recent_errors,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get consecutive errors: {str(e)}"
        )


@router.put("/{hotkey}/pause", response_model=MinerPauseResponse)
async def pause_miner(
    hotkey: str,
    request: MinerPauseRequest,
    config_dao: SystemConfigDAO = Depends(get_system_config_dao),
):
    """
    Pause a miner for a specified duration.
    
    Used by executor when consecutive errors exceed threshold.
    """
    try:
        current_time = int(time.time())
        
        # Default pause duration: 10 minutes
        duration = request.duration_seconds or 600
        paused_until = current_time + duration
        
        # Store pause status in system_config
        pause_key = f"miner_pause:{hotkey}"
        await config_dao.set_param(
            param_name=pause_key,
            param_value={
                "paused_until": paused_until,
                "reason": request.reason,
                "paused_at": current_time,
            },
            param_type="dict",
            description=f"Pause status for miner {hotkey}",
            updated_by="executor",
        )
        
        return MinerPauseResponse(
            miner_hotkey=hotkey,
            is_paused=True,
            pause_until=paused_until,
            reason=request.reason,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause miner: {str(e)}"
        )


@router.get("/{hotkey}/stats", response_model=MinerStatsResponse, dependencies=[Depends(rate_limit_read)])
async def get_miner_stats(
    hotkey: str,
    model_revision: str = Query(..., description="Model revision"),
    env: str = Query(..., description="Environment name"),
    samples_dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Get miner statistics for completion round detection.
    
    Returns unique task_ids to check if miner has completed a full round
    of sampling for the specified environment.
    """
    try:
        # Get all samples for this miner+env
        samples = await samples_dao.get_samples_by_miner(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            include_extra=False,  # Only need task_id
        )
        
        # Extract unique task_ids (numeric part)
        task_ids = set()
        for sample in samples:
            tid = sample.get("task_id", "")
            # Parse task_id format: "timestamp-env-task_id"
            if "-" in tid:
                parts = tid.split("-")
                if len(parts) >= 3:
                    try:
                        task_ids.add(int(parts[-1]))
                    except ValueError:
                        continue
            else:
                try:
                    task_ids.add(int(tid))
                except ValueError:
                    continue
        
        # Get dataset length
        dataset_len = DATASET_LENGTHS.get(env, 200)
        
        # Calculate completion percentage
        completion_pct = (len(task_ids) / dataset_len * 100) if dataset_len > 0 else 0.0
        
        return MinerStatsResponse(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            total_samples=len(samples),
            unique_task_ids=sorted(list(task_ids)),
            dataset_length=dataset_len,
            completion_percentage=completion_pct,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get miner stats: {str(e)}"
        )