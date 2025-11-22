"""
Sample Results Router

Endpoints for submitting and querying sample results.
"""

import time
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from affine.api.models import (
    SampleSubmitResponse,
    SampleListResponse,
    SampleFullResponse,
    SampleDetail,
    PaginationInfo,
)
from affine.api.dependencies import (
    get_sample_results_dao,
    get_miners_dao,
    get_task_pool_manager,
    verify_executor_auth,
    rate_limit_read,
    rate_limit_write,
    rate_limit_scoring,
)
from affine.api.config import config
from affine.core.models import SampleSubmission
from affine.api.utils.pagination import get_pagination_params, create_pagination_info
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.miners import MinersDAO
from affine.api.services.task_pool import TaskPoolManager
from affine.api.utils.bittensor import get_subtensor

router = APIRouter(prefix="/samples", tags=["Samples"])


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
        # Query all samples for this miner+revision+env and filter by task_id
        # Use reverse=True to get newest first, then filter for matching task_id
        samples = await dao.get_samples_by_miner(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            reverse=True,
            include_extra=True
        )
        
        # Filter for matching task_id and get the latest one
        matching_samples = [s for s in samples if str(s.get('task_id')) == str(task_id)]
        
        if not matching_samples:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample not found for hotkey={hotkey}, env={env}, task_id={task_id}"
            )
        
        # Get the latest sample (first in filtered list due to reverse=True sort)
        item = matching_samples[0]
        
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
        
        # Query all samples for this miner+revision+env and filter by task_id
        samples = await sample_dao.get_samples_by_miner(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            reverse=True,
            include_extra=True
        )
        
        # Filter for matching task_id and get the latest one
        matching_samples = [s for s in samples if str(s.get('task_id')) == str(task_id)]
        
        if not matching_samples:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample not found for UID={uid}, env={env}, task_id={task_id}"
            )
        
        # Get the latest sample (first in filtered list due to reverse=True sort)
        item = matching_samples[0]
        
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