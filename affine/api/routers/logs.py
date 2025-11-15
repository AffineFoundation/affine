"""
Execution Logs Router

Endpoints for querying execution logs and error tracking.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from affine.api.models import (
    ExecutionLogsResponse,
    ExecutionLog,
    ConsecutiveErrorsResponse,
    RecentError,
)
from affine.api.dependencies import (
    get_execution_logs_dao,
    rate_limit_read,
)
from affine.database.dao.execution_logs import ExecutionLogsDAO

router = APIRouter(prefix="/logs", tags=["Logs"])


@router.get("/miner/{hotkey}", response_model=ExecutionLogsResponse, dependencies=[Depends(rate_limit_read)])
async def get_miner_logs(
    hotkey: str,
    limit: int = Query(100, description="Maximum number of logs", ge=1, le=1000),
    success: Optional[bool] = Query(None, description="Filter by success/failure"),
    dao: ExecutionLogsDAO = Depends(get_execution_logs_dao),
):
    """
    Get recent execution logs for a miner.
    
    Query parameters:
    - limit: Maximum logs to return (default: 100, max: 1000)
    - success: Filter by success (true) or failure (false) (optional)
    
    Note: model_revision filtering not supported by DAO
    """
    try:
        # Determine status filter based on success parameter
        status_filter = None
        if success is not None:
            status_filter = 'success' if success else 'failed'
        
        # DAO only accepts miner_hotkey, limit, and status
        logs = await dao.get_recent_logs(
            miner_hotkey=hotkey,
            limit=limit,
            status=status_filter,
        )
        
        log_entries = [
            ExecutionLog(
                log_id=log["log_id"],
                timestamp=log["timestamp"],
                task_id=log["task_id"],
                env=log["env"],
                success=(log["status"] == "success"),
                error_type=log.get("error_type"),
                error_message=log.get("error_message"),
                latency_ms=log.get("execution_time_ms", 0),
            )
            for log in logs
        ]
        
        return ExecutionLogsResponse(logs=log_entries)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve logs: {str(e)}"
        )


@router.get("/miner/{hotkey}/consecutive-errors", response_model=ConsecutiveErrorsResponse, dependencies=[Depends(rate_limit_read)])
async def check_consecutive_errors(
    hotkey: str,
    threshold: int = Query(10, description="Error threshold for pausing", ge=1),
    dao: ExecutionLogsDAO = Depends(get_execution_logs_dao),
):
    """
    Check consecutive errors for a miner.
    
    Returns information about consecutive errors and whether the miner should be paused.
    
    Query parameters:
    - threshold: Number of consecutive errors before recommending pause (default: 10)
    
    Note: model_revision filtering not supported by DAO
    """
    try:
        # DAO only accepts miner_hotkey and threshold
        has_consecutive_errors = await dao.check_consecutive_errors(
            miner_hotkey=hotkey,
            threshold=threshold,
        )
        
        # Get recent error details
        recent_error_logs = await dao.get_recent_logs(
            miner_hotkey=hotkey,
            limit=threshold,
            status='failed',
        )
        
        # Count consecutive errors from most recent
        consecutive_errors = 0
        for log in recent_error_logs:
            if log.get('status') == 'failed':
                consecutive_errors += 1
            else:
                break
        
        recent_errors = [
            RecentError(
                timestamp=log["timestamp"],
                error_type=log.get("error_type", "UNKNOWN"),
                error_message=log.get("error_message", ""),
            )
            for log in recent_error_logs
        ]
        
        return ConsecutiveErrorsResponse(
            miner_hotkey=hotkey,
            consecutive_errors=consecutive_errors,
            threshold=threshold,
            should_pause=(consecutive_errors >= threshold),
            recent_errors=recent_errors,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check consecutive errors: {str(e)}"
        )