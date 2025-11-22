"""
Task Queue Router

Endpoints for managing sampling tasks with weighted random selection.
"""

import time
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query, status
from affine.api.models import (
    TaskFetchResponse,
)
from affine.api.dependencies import (
    get_task_queue_dao,
    verify_executor_auth,
    rate_limit_read,
    rate_limit_write,
)
from affine.api.config import config
from affine.database.dao.task_queue import TaskQueueDAO
from affine.api.services.task_pool import TaskPoolManager

from affine.core.setup import logger

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.post("/fetch", response_model=TaskFetchResponse, dependencies=[Depends(rate_limit_read)])
async def fetch_task(
    env: Optional[str] = Query(None, description="Environment filter (optional)"),
    executor_hotkey: str = Depends(verify_executor_auth),
):
    """
    Fetch a task using weighted random selection.
    
    Algorithm:
    1. Verify executor signature and validator status (via dependency)
    2. Get all pending tasks (excluding locked ones)
    3. Group by (miner_hotkey, model_revision), count tasks per miner
    4. Select miner with probability proportional to task count
    5. Randomly select one task from chosen miner
    6. Acquire in-memory lock for selected task
    
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
        logger.info(f"Task fetch requested by executor {executor_hotkey[:16]}... for env {env or 'any'}")
        
        # Fetch task using TaskPoolManager
        pool_manager = TaskPoolManager.get_instance()
        task = await pool_manager.fetch_task(
            executor_hotkey=executor_hotkey,
            env=env
        )
        
        if not task:
            logger.debug(f"No available tasks for executor {executor_hotkey[:16]}...")
            return TaskFetchResponse(task=None)
        
        # Return task details
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
