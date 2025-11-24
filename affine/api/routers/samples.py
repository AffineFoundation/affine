"""
Sample Results Router

Endpoints for querying sample results.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from affine.api.models import (
    SampleFullResponse,
)
from affine.api.dependencies import (
    get_sample_results_dao,
    get_miners_dao,
    get_task_pool_manager,
    rate_limit_read,
    rate_limit_scoring,
)
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.miners import MinersDAO
from affine.api.services.task_pool import TaskPoolManager

router = APIRouter(prefix="/samples", tags=["Samples"])


@router.get("/{hotkey}/{env}/{task_id}", response_model=SampleFullResponse, dependencies=[Depends(rate_limit_read)])
async def get_sample(
    hotkey: str,
    env: str,
    task_id: str,
    model_revision: str = Query(..., description="Model revision"),
    dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Get a specific sample by its natural key components.
    
    Path parameters:
    - hotkey: Miner hotkey
    - env: Environment (e.g., affine:sat)
    - task_id: Task identifier
    
    Query parameters:
    - model_revision: Model revision hash
    
    Returns full sample details including conversation data.
    If multiple submissions exist for the same task_id, returns the latest one by timestamp.
    """
    try:
        # Direct key lookup - O(1) operation
        item = await dao.get_sample_by_task_id(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            task_id=task_id,
            include_extra=True
        )
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample not found for hotkey={hotkey}, env={env}, task_id={task_id}"
            )
        
        return SampleFullResponse(
            miner_hotkey=item["miner_hotkey"],
            model_revision=item["model_revision"],
            env=item["env"],
            score=item["score"],
            signature=item["signature"],
            extra=item.get("extra", {}),
            timestamp=item["timestamp"],
            block_number=item["block_number"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sample: {str(e)}"
        )


@router.get("/uid/{uid}/{env}/{task_id}", response_model=SampleFullResponse, dependencies=[Depends(rate_limit_read)])
async def get_sample_by_uid(
    uid: int,
    env: str,
    task_id: str,
    sample_dao: SampleResultsDAO = Depends(get_sample_results_dao),
    miners_dao: MinersDAO = Depends(get_miners_dao),
):
    """
    Get a specific sample by UID, env, and task_id.
    
    Path parameters:
    - uid: Miner UID (0-255)
    - env: Environment (e.g., affine:sat)
    - task_id: Task identifier
    
    Returns full sample details including conversation data.
    Automatically looks up the miner's current hotkey and revision.
    If multiple submissions exist for the same task_id, returns the latest one by timestamp.
    """
    try:
        # Get miner info by UID
        miner = await miners_dao.get_miner_by_uid(uid)
        
        if not miner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner not found for UID={uid}"
            )
        
        # Extract hotkey and revision
        hotkey = miner['hotkey']
        model_revision = miner['revision']
        
        # Direct key lookup - O(1) operation
        item = await sample_dao.get_sample_by_task_id(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            task_id=task_id,
            include_extra=True
        )
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample not found for UID={uid}, env={env}, task_id={task_id}"
            )
        
        return SampleFullResponse(
            miner_hotkey=item["miner_hotkey"],
            model_revision=item["model_revision"],
            env=item["env"],
            score=item["score"],
            signature=item["signature"],
            extra=item.get("extra", {}),
            timestamp=item["timestamp"],
            block_number=item["block_number"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sample: {str(e)}"
        )


@router.get("/pool/uid/{uid}/{env}", dependencies=[Depends(rate_limit_read)])
async def get_task_pool(
    uid: int,
    env: str,
    miners_dao: MinersDAO = Depends(get_miners_dao),
    task_pool: TaskPoolManager = Depends(get_task_pool_manager),
):
    """
    Get pending task IDs for a specific miner in an environment.
    
    Path parameters:
    - uid: Miner UID (0-255)
    - env: Environment (e.g., agentgym:webshop)
    
    Returns list of pending task_ids in the queue for this miner.
    """
    try:
        # Get miner info by UID
        miner = await miners_dao.get_miner_by_uid(uid)
        
        if not miner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Miner not found for UID={uid}"
            )
        
        # Extract hotkey and revision
        hotkey = miner['hotkey']
        model_revision = miner['revision']
        
        # Get tasks for this miner in the specified environment
        tasks = await task_pool.dao.get_tasks_by_miner(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env
        )
        
        # Extract task_ids from pending tasks
        task_ids = [
            task['task_id']
            for task in tasks
            if task.get('status') in ['pending', 'assigned']
        ]
        
        return {
            "uid": uid,
            "hotkey": hotkey,
            "model_revision": model_revision,
            "env": env,
            "task_count": len(task_ids),
            "task_ids": sorted(task_ids)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve task pool: {str(e)}"
        )


@router.get("/scoring", dependencies=[Depends(rate_limit_scoring)])
async def get_scoring_data():
    """
    Get scoring data for all valid miners.
    
    Uses proactive cache with background refresh.
    - Startup: Cache prewarmed
    - Runtime: Background refresh every 20 minutes
    - Access: Always returns hot cache (< 100ms)
    """
    from affine.api.services.scoring_cache import get_cached_data
    
    try:
        return await get_cached_data()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scoring data: {str(e)}"
        )