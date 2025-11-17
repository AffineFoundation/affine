"""
Task Queue Router

Endpoints for managing sampling tasks.
"""

import time
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, Query, status
from affine.api.models import (
    TaskCreateRequest,
    TaskCreateResponse,
    TaskListResponse,
    TaskDetail,
    TaskStatusResponse,
    TaskCompleteRequest,
    TaskFailRequest,
    TaskQueueStatsResponse,
    TaskFetchRequest,
)
from affine.api.dependencies import (
    get_task_queue_dao,
    get_sample_results_dao,
    get_auth_service,
    get_task_generator_service,
    rate_limit_read,
    rate_limit_write,
)
from affine.database.dao.task_queue import TaskQueueDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.api.services.auth import AuthService
from affine.api.services.task_generator import TaskGeneratorService

logger = logging.getLogger(__name__)

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


@router.get("/queue-stats", response_model=TaskQueueStatsResponse, dependencies=[Depends(rate_limit_read)])
async def get_queue_stats(
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
):
    """
    Get task queue statistics.
    
    Returns pending task counts by environment and overall stats.
    """
    try:
        # Get all pending tasks
        pending_tasks = await dao.get_pending_tasks(limit=None)
        
        # Count by environment
        pending_by_env = {}
        running_count = 0
        
        for task in pending_tasks:
            env = task.get("env", "unknown")
            task_status = task.get("status")
            
            if task_status == "pending":
                pending_by_env[env] = pending_by_env.get(env, 0) + 1
            elif task_status == "running":
                running_count += 1
        
        return TaskQueueStatsResponse(
            pending_by_env=pending_by_env,
            total_pending=sum(pending_by_env.values()),
            running_count=running_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue stats: {str(e)}"
        )


@router.get("/fetch", response_model=TaskListResponse, dependencies=[Depends(rate_limit_read)])
async def fetch_tasks(
    env: Optional[str] = Query(None, description="Environment filter"),
    worker_id: str = Query(..., description="Worker ID"),
    limit: int = Query(1, description="Number of tasks to fetch", ge=1, le=10),
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
):
    """
    Atomically fetch tasks for execution.
    
    This endpoint is used by executor workers to get pending tasks.
    Workers should immediately call /start after fetching.
    
    Query parameters:
    - env: Filter by environment (optional)
    - worker_id: Worker identifier
    - limit: Number of tasks to fetch (1-10)
    """
    try:
        # Get pending tasks with priority ordering
        pending_tasks = await dao.get_pending_tasks(limit=limit * 5, priority_order=True)
        
        # Filter by env if specified
        if env:
            pending_tasks = [t for t in pending_tasks if t.get("env") == env]
        
        # Take only requested limit
        pending_tasks = pending_tasks[:limit]
        
        # Convert to TaskDetail format
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
            for t in pending_tasks
        ]
        
        return TaskListResponse(tasks=task_details)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tasks: {str(e)}"
        )


@router.post("/fetch-authenticated", response_model=TaskListResponse, dependencies=[Depends(rate_limit_read)])
async def fetch_tasks_authenticated(
    env: str = Query(..., description="Environment to fetch tasks for"),
    executor_hotkey: str = Header(..., alias="X-Executor-Hotkey", description="Executor's hotkey"),
    executor_signature: str = Header(..., alias="X-Executor-Signature", description="Signed message"),
    executor_message: str = Header(..., alias="X-Executor-Message", description="Original message"),
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Fetch task with executor authentication (signature verification).
    
    This endpoint:
    1. Verifies executor signature
    2. Checks if executor is a registered validator
    3. Returns a task and assigns it to the executor
    4. Validates task is still valid (miner still active)
    
    Headers:
    - X-Executor-Hotkey: Executor's SS58 hotkey
    - X-Executor-Signature: Hex-encoded signature of message
    - X-Executor-Message: Original message that was signed
    
    Query parameters:
    - env: Environment to fetch tasks for
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
        
        # Step 2: Verify executor is a validator (TODO: Check against metagraph)
        # For now, we trust any valid signature
        logger.info(f"Task fetch requested by executor {executor_hotkey[:16]}... for env {env}")
        
        # Step 3: Get next pending task for this environment
        task = await dao.get_next_pending_task(env)
        
        if not task:
            logger.debug(f"No pending tasks for env {env}")
            return TaskListResponse(tasks=[])
        
        # Step 4: Assign task to executor (atomically mark as assigned)
        assigned_task = await dao.assign_task(task, executor_hotkey)
        
        # Step 5: Convert to response format
        # task_id = dataset index (integer as string), task_uuid = queue identifier
        task_detail = TaskDetail(
            task_id=str(assigned_task["task_id"]),  # Dataset index
            miner_hotkey=assigned_task["miner_hotkey"],
            model_revision=assigned_task["model_revision"],
            model=assigned_task["model"],
            env=assigned_task["env"],
            status=assigned_task["status"],
            created_at=assigned_task["created_at"],
            task_uuid=assigned_task["task_uuid"],  # Queue UUID for completion
        )
        
        logger.info(
            f"Assigned task {assigned_task['task_uuid']} "
            f"(dataset_idx={assigned_task['task_id']}) to executor {executor_hotkey[:16]}..."
        )
        
        return TaskListResponse(tasks=[task_detail])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching authenticated task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch task: {str(e)}"
        )


@router.post("/complete-authenticated", response_model=TaskStatusResponse, dependencies=[Depends(rate_limit_write)])
async def complete_task_authenticated(
    task_uuid: str = Query(..., description="Task UUID from queue"),
    dataset_task_id: int = Query(..., description="Dataset task index"),
    success: bool = Query(..., description="Whether task succeeded"),
    error_message: Optional[str] = Query(None, description="Error message if failed"),
    executor_hotkey: str = Header(..., alias="X-Executor-Hotkey", description="Executor's hotkey"),
    executor_signature: str = Header(..., alias="X-Executor-Signature", description="Signed message"),
    executor_message: str = Header(..., alias="X-Executor-Message", description="Original message"),
    dao: TaskQueueDAO = Depends(get_task_queue_dao),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Complete a task with executor authentication.
    
    This endpoint:
    1. Verifies executor signature
    2. Marks task as completed (removes from queue) or failed (increments retry)
    3. Records execution log
    
    Query parameters:
    - task_uuid: UUID of task in queue
    - dataset_task_id: Dataset index of the task
    - success: True if task succeeded
    - error_message: Error message if failed (optional)
    
    Headers:
    - X-Executor-Hotkey: Executor's SS58 hotkey
    - X-Executor-Signature: Hex-encoded signature
    - X-Executor-Message: Original message
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
        
        # Step 2: Get task by UUID
        task = await dao.get_task_by_uuid(task_uuid)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_uuid} not found"
            )
        
        # Verify task was assigned to this executor
        if task.get("assigned_to") != executor_hotkey:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Task not assigned to this executor"
            )
        
        # Step 3: Update task status
        if success:
            # Remove from queue
            await dao.complete_task(task)
            
            logger.info(
                f"Task {task_uuid} (dataset_idx={dataset_task_id}) "
                f"completed successfully by {executor_hotkey[:16]}..."
            )
            
            return TaskStatusResponse(
                task_id=task_uuid,
                status="completed",
                completed_at=int(time.time())
            )
        else:
            # Mark as failed (will retry if under limit)
            updated_task = await dao.fail_task(
                task,
                error_message=error_message or "Unknown error",
                error_code="EXECUTION_ERROR"
            )
            
            logger.warning(
                f"Task {task_uuid} (dataset_idx={dataset_task_id}) "
                f"failed: {error_message}"
            )
            
            return TaskStatusResponse(
                task_id=task_uuid,
                status=updated_task["status"],  # 'pending' if retrying, 'failed' if exceeded
                failed_at=int(time.time())
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete task: {str(e)}"
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
        # Use new get_task_by_id method to find task
        task = await dao.get_task_by_id(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        # Start the task
        success = await dao.start_task(
            task["miner_hotkey"],
            task["model_revision"],
            task_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task status"
            )
        
        return TaskStatusResponse(
            task_id=task_id,
            status="running",
            started_at=int(time.time()),
        )
        
    except HTTPException:
        raise
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
    Mark task as completed and remove from queue.
    
    Requires the sample_id of the completed sample result.
    """
    try:
        # Use get_task_by_id to find task
        task = await dao.get_task_by_id(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        # Complete the task (deletes from queue)
        success = await dao.complete_task(
            task["miner_hotkey"],
            task["model_revision"],
            task_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to complete task"
            )
        
        return TaskStatusResponse(
            task_id=task_id,
            status="completed",
            completed_at=int(time.time()),
        )
        
    except HTTPException:
        raise
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
        # Use get_task_by_id to find task
        task = await dao.get_task_by_id(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        # Fail the task
        success = await dao.fail_task(
            task["miner_hotkey"],
            task["model_revision"],
            task_id,
            data.error_message
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task status"
            )
        
        return TaskStatusResponse(
            task_id=task_id,
            status="failed",
            failed_at=int(time.time()),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark task as failed: {str(e)}"
        )