"""
Task Queue Router

Endpoints for managing sampling tasks.
"""

import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from affine.api.models import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskListResponse,
    TaskDetail,
    TaskStatusResponse,
    TaskCompleteRequest,
    TaskFailRequest,
)
from affine.api.dependencies import (
    get_task_queue_dao,
    rate_limit_read,
    rate_limit_write,
)
from affine.database.dao.task_queue import TaskQueueDAO

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("", response_model=TaskCreateResponse, dependencies=[Depends(rate_limit_write)])
async def create_task(
    task: TaskCreateRequest,
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
):
    """
    Create a new sampling task.
    
    The task will be added to the queue for execution.
    """
    try:
        # Create task using correct DAO signature
        result = await dao.create_task(
            miner_hotkey=task.miner_hotkey,
            model_revision=task.model_revision,
            model=task.model,
            env=task.env,
            validator_hotkey=task.validator_hotkey,
            priority=task.priority if hasattr(task, 'priority') else 0,
            ttl_days=7,
        )
        
        task_id = result['task_id']
        
        return TaskCreateResponse(
            task_id=task_id,
            status="pending",
            created_at=int(time.time()),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )


@router.get("/pending", response_model=TaskListResponse, dependencies=[Depends(rate_limit_read)])
async def get_pending_tasks(
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    model_revision: Optional[str] = Query(None, description="Filter by model revision"),
    limit: int = Query(10, description="Maximum number of tasks", ge=1, le=100),
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
):
    """
    Get pending tasks for execution.
    
    Query parameters:
    - miner_hotkey: Filter by miner (optional)
    - model_revision: Filter by model revision (optional)
    - limit: Maximum results (default: 10, max: 100)
    """
    try:
        # Get pending tasks - pass limit only (priority_order is default True)
        tasks = await dao.get_pending_tasks(limit=limit, priority_order=True)
        
        # Filter by miner_hotkey if specified
        if miner_hotkey:
            tasks = [t for t in tasks if t.get("miner_hotkey") == miner_hotkey]
        
        # Filter by model_revision if specified
        if model_revision:
            tasks = [t for t in tasks if t.get("model_revision") == model_revision]
        
        task_details = [
            TaskDetail(
                task_id=t["task_id"],
                miner_hotkey=t["miner_hotkey"],
                model_revision=t["model_revision"],
                model=t["model"],
                env=t["env"],
                status=t["status"],
                created_at=t["created_at"],
            )
            for t in tasks
        ]
        
        return TaskListResponse(tasks=task_details)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tasks: {str(e)}"
        )


@router.put("/{task_id}/start", response_model=TaskStatusResponse, dependencies=[Depends(rate_limit_write)])
async def start_task(
    task_id: str,
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
):
    """
    Mark task as started.
    
    Updates the task status to 'running' and records the start time.
    """
    try:
        # Need to get task first to extract miner_hotkey and model_revision
        # For now, return error - need task lookup mechanism
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Task operations require storing task_id -> (miner_hotkey, model_revision) mapping"
        )
        
        # Once we have mapping:
        # await dao.start_task(miner_hotkey, model_revision, task_id)
        
        return TaskStatusResponse(
            task_id=task_id,
            status="running",
            started_at=int(time.time()),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start task: {str(e)}"
        )


@router.put("/{task_id}/complete", response_model=TaskStatusResponse, dependencies=[Depends(rate_limit_write)])
async def complete_task(
    task_id: str,
    data: TaskCompleteRequest,
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
):
    """
    Mark task as completed.
    
    Requires the sample_id of the completed sample result.
    """
    try:
        # Need task lookup - see start_task issue
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Task operations require storing task_id -> (miner_hotkey, model_revision) mapping"
        )
        
        # Once we have mapping:
        # await dao.complete_task(miner_hotkey, model_revision, task_id)
        
        return TaskStatusResponse(
            task_id=task_id,
            status="completed",
            completed_at=int(time.time()),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete task: {str(e)}"
        )


@router.put("/{task_id}/fail", response_model=TaskStatusResponse, dependencies=[Depends(rate_limit_write)])
async def fail_task(
    task_id: str,
    data: TaskFailRequest,
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
):
    """
    Mark task as failed.
    
    Records the error type and message, and increments the error count.
    """
    try:
        # Need task lookup - see start_task issue
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Task operations require storing task_id -> (miner_hotkey, model_revision) mapping"
        )
        
        # Once we have mapping:
        # await dao.fail_task(miner_hotkey, model_revision, task_id, data.error_message)
        
        return TaskStatusResponse(
            task_id=task_id,
            status="failed",
            failed_at=int(time.time()),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark task as failed: {str(e)}"
        )