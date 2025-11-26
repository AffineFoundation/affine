"""
Task Queue Router

Endpoints for managing sampling tasks with weighted random selection.
"""

import time
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query, status
from affine.api.models import (
    TaskFetchResponse,
    SampleSubmitResponse,
)
from affine.api.dependencies import (
    get_sample_results_dao,
    get_task_pool_manager,
    verify_executor_auth,
    rate_limit_read,
    get_miners_dao,
)
from affine.api.config import config
from affine.core.models import SampleSubmission
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.miners import MinersDAO
from affine.api.services.task_pool import TaskPoolManager
from affine.utils.subtensor import get_subtensor

from affine.core.setup import logger

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("/fetch", response_model=TaskFetchResponse)
async def fetch_task(
    env: Optional[str] = Query(None, description="Environment filter (optional)"),
    executor_hotkey: str = Depends(verify_executor_auth),
    miners_dao: MinersDAO = Depends(get_miners_dao),
    task_pool: TaskPoolManager = Depends(get_task_pool_manager),
):
    """
    Fetch a task using weighted random selection.
    
    Algorithm:
    1. Verify executor signature and validator status (via dependency)
    2. Get all pending tasks
    3. Group by (miner_hotkey, model_revision), count tasks per miner
    4. Select miner with probability proportional to task count
    5. Randomly select one task from chosen miner
    6. Assign task (DynamoDB provides atomicity)
    
    Headers (validated by verify_executor_auth dependency):
    - X-Hotkey: Executor's SS58 hotkey
    - X-Signature: Hex-encoded signature of timestamp
    - X-Message: Unix timestamp (must be within 60 seconds)
    
    Query Parameters:
    - env: Optional environment filter
    
    Returns:
    - Task details if available, or null if no tasks
    """
    # Check if services are enabled
    if not config.SERVICES_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task fetch service is disabled (SERVICES_ENABLED=false)"
        )
    
    try:
        # Fetch task using TaskPoolManager (injected via dependency)
        task = await task_pool.fetch_task(
            executor_hotkey=executor_hotkey,
            env=env
        )
        
        if not task:
            logger.debug(f"No available tasks for executor {executor_hotkey[:16]}...")
            return TaskFetchResponse(task=None)
        
        miner_record = await miners_dao.get_miner_by_hotkey(task["miner_hotkey"])
        if not miner_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner record not found for hotkey {task['miner_hotkey'][:16]}..."
            )
        
        miner_uid = miner_record.get("uid")
        if miner_uid is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"UID not found for hotkey {task['miner_hotkey'][:16]}..."
            )
        
        # Return task details
        logger.debug(
            f"Assigned task {task['task_uuid']} to executor {executor_hotkey[:16]}... "
            f"(miner={task['miner_hotkey'][:16]}..., uid={miner_uid}, env={task['env']}, task_id={task['task_id']})"
        )
        
        return TaskFetchResponse(
            task={
                "task_uuid": task["task_uuid"],
                "task_id": task["task_id"],
                "miner_hotkey": task["miner_hotkey"],
                "miner_uid": miner_uid,
                "model_revision": task["model_revision"],
                "model": task["model"],
                "env": task["env"],
                "chute_id": task["chute_id"],
                "created_at": task["created_at"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch task: {str(e)}"
        )


@router.post("/submit", response_model=SampleSubmitResponse)
async def submit_sample_from_executor(
    submission: Dict[str, Any],
    executor_hotkey: str = Depends(verify_executor_auth),
    sample_dao: SampleResultsDAO = Depends(get_sample_results_dao),
    task_pool: TaskPoolManager = Depends(get_task_pool_manager),
):
    """
    Submit a sample result from executor.
    
    This endpoint:
    1. Verifies executor authentication via dependency (timestamp-based)
    2. Validates submission signature against task_uuid data
    3. Saves sample to database (if successful)
    4. Completes task via TaskPoolManager (releases lock, logs execution, deletes task)
    
    Headers (validated by verify_executor_auth dependency):
    - X-Hotkey: Executor's SS58 hotkey
    - X-Signature: Hex-encoded signature of timestamp
    - X-Message: Unix timestamp (must be within 60 seconds)
    
    Request body (SampleSubmission):
    - task_uuid: Task UUID from queue
    - score: Evaluation score (0.0 to 1.0)
    - latency_ms: Execution time in milliseconds
    - extra: Evaluation details and metadata
    - signature: Executor's signature of the above fields
    """
    # Check if services are enabled
    if not config.SERVICES_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sample submit service is disabled (SERVICES_ENABLED=false)"
        )
    
    # Parse submission
    try:
        sample_sub = SampleSubmission(**submission)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid submission format: {str(e)}"
        )
    
    # Verify submission signature
    if not sample_sub.verify(executor_hotkey):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid submission signature"
        )
    
    # Determine task outcome based on error presence
    error_message = sample_sub.extra.get("error")
    is_success = error_message is None
    
    # Save sample if task succeeded
    if is_success:
        # Get task to extract metadata
        task = await task_pool.dao.get_task_by_uuid(sample_sub.task_uuid)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {sample_sub.task_uuid}"
            )
        
        # Get current block number
        subtensor = await get_subtensor()
        block_number = await subtensor.get_current_block()
        
        # Save sample
        try:
            await sample_dao.save_sample(
                miner_hotkey=task["miner_hotkey"],
                model_revision=task["model_revision"],
                model=task["model"],
                env=task["env"],
                task_id=str(task["task_id"]),
                score=sample_sub.score,
                latency_ms=sample_sub.latency_ms,
                extra=sample_sub.extra,
                validator_hotkey=executor_hotkey,
                block_number=block_number,
                signature=sample_sub.signature,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save sample: {str(e)}"
            )
    
    # Complete task via TaskPoolManager (handles lock release, logging, and deletion/retry)
    result = await task_pool.complete_task(
        task_uuid=sample_sub.task_uuid,
        executor_hotkey=executor_hotkey,
        success=is_success,
        error_message=error_message,
        error_code="EXECUTION_ERROR" if error_message else None
    )
    
    # Build response message
    if result['status'] == 'completed':
        message = f"Sample submitted successfully (score={sample_sub.score:.4f})"
    elif result['status'] == 'not_found':
        message = "Task already completed or removed"
    elif result['status'] == 'failed':
        message = result.get('message', 'Task failed')
    else:
        message = result.get('message', 'Task processing completed')
    
    return SampleSubmitResponse(
        task_id=sample_sub.task_uuid,
        created_at=int(time.time()),
        message=message
    )

@router.get("/pool/stats", dependencies=[Depends(rate_limit_read)])
async def get_pool_stats(
    env: Optional[str] = Query(None, description="Environment filter (optional)"),
    task_pool: TaskPoolManager = Depends(get_task_pool_manager),
):
    """
    Get task queue statistics for monitoring.
    
    Query Parameters:
    - env: Optional environment filter (e.g., "agentgym:alfworld")
    
    Returns:
    - pending_count: Number of pending tasks in the queue
    - assigned_count: Number of assigned tasks
    - env: Environment name (if filtered)
    """
    try:
        if env:
            # Get stats using efficient COUNT query
            stats = await task_pool.dao.get_pool_stats(env)
            
            return {
                "env": env,
                "pending_count": stats.get('pending', 0),
                "assigned_count": stats.get('assigned', 0),
                "failed_count": stats.get('failed', 0),
            }
        else:
            # Get total stats across all environments
            # This would require querying all environments
            # For now, return error asking for env parameter
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="env parameter is required"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue stats: {str(e)}"
        )
