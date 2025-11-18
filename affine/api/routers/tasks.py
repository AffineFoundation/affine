"""
Task Queue Router

Endpoints for managing sampling tasks with weighted random selection.
"""

import time
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query, status
from affine.api.models import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskFetchResponse,
    TaskCompleteRequest,
    TaskCompleteResponse,
    TaskPoolStatsResponse,
)
from affine.api.dependencies import (
    get_task_queue_dao,
    get_auth_service,
    rate_limit_read,
    rate_limit_write,
)
from affine.database.dao.task_queue import TaskQueueDAO
from affine.api.services.auth import AuthService
from affine.api.services.task_pool import TaskPoolManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("/fetch", response_model=TaskFetchResponse, dependencies=[Depends(rate_limit_read)])
async def fetch_task(
    env: Optional[str] = Query(None, description="Environment filter (optional)"),
    executor_hotkey: str = Header(..., alias="X-Executor-Hotkey", description="Executor's hotkey"),
    executor_signature: str = Header(..., alias="X-Executor-Signature", description="Signed message"),
    executor_message: str = Header(..., alias="X-Executor-Message", description="Original message"),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Fetch a task using weighted random selection.
    
    Algorithm:
    1. Verify executor signature and validator status
    2. Get all pending tasks (excluding locked ones)
    3. Group by (miner_hotkey, model_revision), count tasks per miner
    4. Select miner with probability proportional to task count
    5. Randomly select one task from chosen miner
    6. Acquire in-memory lock for selected task
    
    Headers:
    - X-Executor-Hotkey: Executor's SS58 hotkey
    - X-Executor-Signature: Hex-encoded signature of message
    - X-Executor-Message: Original message that was signed
    
    Query Parameters:
    - env: Optional environment filter
    
    Returns:
    - Task details if available, or null if no tasks
    """
    try:
        # Step 1: Verify signature
        is_valid = auth_service.verify_signature(
            hotkey=executor_hotkey,
            message=executor_message,
            signature=executor_signature
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid executor signature"
            )
        
        # Step 2: Verify executor is a validator (TODO: Check metagraph)
        # For now, trust any valid signature
        logger.info(f"Task fetch requested by executor {executor_hotkey[:16]}... for env {env or 'any'}")
        
        # Step 3: Fetch task using TaskPoolManager
        pool_manager = TaskPoolManager.get_instance()
        task = await pool_manager.fetch_task(
            executor_hotkey=executor_hotkey,
            env=env
        )
        
        if not task:
            logger.debug(f"No available tasks for executor {executor_hotkey[:16]}...")
            return TaskFetchResponse(task=None)
        
        # Step 4: Return task details
        logger.info(
            f"Assigned task {task['task_uuid']} to executor {executor_hotkey[:16]}... "
            f"(miner={task['miner_hotkey'][:16]}..., env={task['env']}, task_id={task['task_id']})"
        )
        
        return TaskFetchResponse(
            task={
                "task_uuid": task["task_uuid"],
                "task_id": task["task_id"],
                "miner_hotkey": task["miner_hotkey"],
                "model_revision": task["model_revision"],
                "model": task["model"],
                "env": task["env"],
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


@router.post("/complete", response_model=TaskCompleteResponse, dependencies=[Depends(rate_limit_write)])
async def complete_task(
    request: TaskCompleteRequest,
    executor_hotkey: str = Header(..., alias="X-Executor-Hotkey", description="Executor's hotkey"),
    executor_signature: str = Header(..., alias="X-Executor-Signature", description="Signed message"),
    executor_message: str = Header(..., alias="X-Executor-Message", description="Original message"),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Complete a task (success or failure).
    
    Idempotent: if task already completed/deleted, returns success status.
    
    Headers:
    - X-Executor-Hotkey: Executor's SS58 hotkey
    - X-Executor-Signature: Hex-encoded signature
    - X-Executor-Message: Original message
    
    Request Body:
    - task_uuid: Task UUID from fetch response
    - success: Whether task succeeded
    - error_message: Error message if failed (optional)
    - error_code: Error classification code (optional)
    
    Returns:
    - Status of completion (completed/failed/not_found/error)
    """
    try:
        # Step 1: Verify signature
        is_valid = auth_service.verify_signature(
            hotkey=executor_hotkey,
            message=executor_message,
            signature=executor_signature
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid executor signature"
            )
        
        # Step 2: Complete task using TaskPoolManager
        pool_manager = TaskPoolManager.get_instance()
        result = await pool_manager.complete_task(
            task_uuid=request.task_uuid,
            executor_hotkey=executor_hotkey,
            success=request.success,
            error_message=request.error_message,
            error_code=request.error_code,
        )
        
        # Step 3: Return result
        status_code = status.HTTP_200_OK
        if result['status'] == 'error':
            if 'Lock owned by different executor' in result['message']:
                status_code = status.HTTP_403_FORBIDDEN
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        return TaskCompleteResponse(
            task_uuid=request.task_uuid,
            status=result['status'],
            message=result['message'],
            timestamp=int(time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete task: {str(e)}"
        )


@router.post("/batch", dependencies=[Depends(rate_limit_write)])
async def batch_create_tasks(
    request: Dict[str, Any],
    dao: TaskQueueDAO = Depends(get_task_queue_dao)
):
    """
    Batch create tasks (used by task generator/scheduler).
    
    Request Body:
    - tasks: List of task dicts with keys:
        - miner_hotkey
        - model_revision
        - model
        - env
        - task_id (integer)
    
    Returns:
    - added_count: Number of tasks successfully created
    """
    try:
        tasks = request.get("tasks", [])
        
        if not tasks:
            return {"added_count": 0}
        
        # Batch create tasks using DAO
        count = await dao.batch_create_tasks(tasks)
        
        logger.info(f"Batch created {count} tasks")
        
        return {"added_count": count}
        
    except Exception as e:
        logger.error(f"Error batch creating tasks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch create tasks: {str(e)}"
        )


@router.get("/queued-task-ids", dependencies=[Depends(rate_limit_read)])
async def get_queued_task_ids(
    miner_hotkey: str = Query(..., description="Miner's hotkey"),
    model_revision: str = Query(..., description="Model revision"),
    env: str = Query(..., description="Environment name"),
    dao: TaskQueueDAO = Depends(get_task_queue_dao)
):
    """
    Get queued task IDs for a miner+revision+env combination.
    
    Used by task generator to check which tasks are already queued.
    
    Query Parameters:
    - miner_hotkey: Miner's hotkey
    - model_revision: Model revision
    - env: Environment name
    
    Returns:
    - task_ids: List of queued task IDs (integers)
    """
    try:
        task_ids = await dao.get_pending_task_ids_for_miner(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=env
        )
        
        return {"task_ids": sorted(list(task_ids))}
        
    except Exception as e:
        logger.error(f"Error getting queued task IDs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queued task IDs: {str(e)}"
        )


@router.get("/pool-stats", response_model=TaskPoolStatsResponse, dependencies=[Depends(rate_limit_read)])
async def get_pool_stats():
    """
    Get task pool statistics.
    
    Returns:
    - Statistics by environment
    - Total pending/locked/assigned/failed counts
    - Lock details (task UUID, executor, lock time, expiration)
    """
    try:
        pool_manager = TaskPoolManager.get_instance()
        stats = await pool_manager.get_pool_stats()
        
        return TaskPoolStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting pool stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pool stats: {str(e)}"
        )