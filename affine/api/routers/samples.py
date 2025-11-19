"""
Sample Results Router

Endpoints for submitting and querying sample results.
"""

import time
import uuid
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request, Query, status
from affine.api.models import (
    SampleSubmitRequest,
    SampleSubmitResponse,
    SampleListResponse,
    SampleFullResponse,
    SampleDetail,
    PaginationInfo,
)
from affine.api.dependencies import (
    get_sample_results_dao,
    get_task_queue_dao,
    verify_signature_dependency,
    verify_executor_auth,
    rate_limit_read,
    rate_limit_write,
)
from affine.core.models import SampleSubmission
from affine.api.utils.pagination import get_pagination_params, create_pagination_info
from affine.database.dao.sample_results import SampleResultsDAO
from affine.core.miners import miners as get_miners
from affine.api.utils.bittensor import get_subtensor

router = APIRouter(prefix="/samples", tags=["Samples"])


@router.post("/submit", response_model=SampleSubmitResponse, dependencies=[Depends(rate_limit_write)])
async def submit_sample_from_executor(
    submission: Dict[str, Any],
    executor_hotkey: str = Depends(verify_executor_auth),
    sample_dao: SampleResultsDAO = Depends(get_sample_results_dao),
    task_dao = Depends(get_task_queue_dao),
):
    """
    Submit a sample result from executor.
    
    This endpoint:
    1. Verifies executor authentication via dependency (timestamp-based)
    2. Validates submission signature against task_uuid data
    3. Fetches task metadata from task queue
    4. Verifies executor matches task assignment
    5. Saves complete sample to database
    6. Marks task as completed in queue
    
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
    
    # Fetch task from queue by UUID
    task = await task_dao.get_task_by_uuid(sample_sub.task_uuid)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {sample_sub.task_uuid}"
        )
    
    # Verify executor matches task assignment
    assigned_executor = task.get("assigned_to")
    if assigned_executor and assigned_executor != executor_hotkey:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Task was assigned to different executor: {assigned_executor}"
        )
    
    # Extract task metadata
    miner_hotkey = task["miner_hotkey"]
    model_revision = task["model_revision"]
    model = task["model"]
    env = task["env"]
    task_id = str(task["task_id"])
    
    # Determine if task succeeded or failed based on error presence
    # Success = task completed (no error in extra), regardless of score
    # Failure = task failed (error in extra), needs retry
    has_error = "error" in sample_sub.extra
    subtensor = await get_subtensor()    
    block_number = await subtensor.get_current_block()

    if not has_error:
        # Task succeeded: save sample and complete task (score can be any value)
        try:
            await sample_dao.save_sample(
                miner_hotkey=miner_hotkey,
                model_revision=model_revision,
                model=model,
                env=env,
                task_id=task_id,
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
        
        # Mark task as completed
        await task_dao.complete_task(task)
    else:
        # Task failed: handle retry logic
        error_message = sample_sub.extra.get("error", "Unknown error")
        
        # Call fail_task to increment retry_count and handle retry/permanent failure
        # fail_task will:
        # - If retry_count < max_retries: reset to pending (no sample saved)
        # - If retry_count >= max_retries: create zero-score sample and mark as failed
        await task_dao.fail_task(
            task=task,
            error_message=error_message,
            error_code="EXECUTION_ERROR"
        )
    
    # Build response message based on task outcome
    if not has_error:
        message = f"Sample submitted successfully (score={sample_sub.score:.4f})"
    else:  # is_failure
        retry_count = task.get('retry_count') + 1
        max_retries = task.get('max_retries')
        if retry_count >= max_retries:
            message = f"Task permanently failed after {retry_count} retries, zero-score sample saved"
        else:
            message = f"Task failed (retry {retry_count}/{max_retries}), re-queued for retry"
    
    return SampleSubmitResponse(
        task_id=task_id,
        created_at=int(time.time()),
        message=message
    )


@router.post("", response_model=SampleSubmitResponse, dependencies=[Depends(rate_limit_write)])
async def submit_sample(
    request: Request,
    sample: SampleSubmitRequest,
    dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Submit a sample result (requires signature verification).
    
    The request must include:
    - X-Hotkey header: Miner's hotkey
    - X-Signature header: Signature of the request body
    
    The signature should be generated by signing the JSON payload with the miner's private key.
    """
    # Verify signature
    hotkey = await verify_signature_dependency(request)
    
    # Verify hotkey matches the sample
    if hotkey != sample.miner_hotkey:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hotkey in signature does not match miner_hotkey in payload"
        )
    
    # Save sample to database
    try:
        await dao.save_sample(
            miner_hotkey=sample.miner_hotkey,
            model_revision=sample.model_revision,
            model=sample.model,
            env=sample.env,
            task_id=sample.task_id,
            score=sample.score,
            latency_ms=sample.latency_ms,
            extra=sample.extra.model_dump(),
            validator_hotkey=sample.validator_hotkey,
            block_number=sample.block_number,
            signature=request.headers.get("X-Signature", ""),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save sample: {str(e)}"
        )
    
    return SampleSubmitResponse(
        task_id=sample.task_id,  # Return task_id as identifier
        created_at=int(time.time()),
        message="Sample saved successfully"
    )


@router.get("/miner/{hotkey}", response_model=SampleListResponse, dependencies=[Depends(rate_limit_read)])
async def get_miner_samples(
    hotkey: str,
    model_revision: str = Query(..., description="Model revision (required)"),
    env: str = Query(..., description="Environment name (required, e.g., affine:sat)"),
    task_id_start: Optional[int] = Query(None, description="Task ID range start (inclusive)"),
    task_id_end: Optional[int] = Query(None, description="Task ID range end (exclusive)"),
    deduplicate: bool = Query(True, description="Deduplicate by task_id, keeping latest sample"),
    limit: int = Query(1000, description="Maximum number of results", ge=1, le=10000),
    dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Get samples for a specific miner, revision, and environment.
    
    Required query parameters:
    - model_revision: Model revision hash
    - env: Environment name (e.g., affine:sat, agentgym:webshop)
    
    Optional query parameters:
    - task_id_start: Start of task ID range (inclusive)
    - task_id_end: End of task ID range (exclusive)
    - deduplicate: If True (default), keep only latest sample per task_id
    - limit: Maximum results to return (default: 1000, max: 10000)
    
    Note: env is required because it's part of the partition key.
    """
    try:
        # Get samples from DAO
        samples = await dao.get_samples_by_miner(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            env=env,
            limit=None,  # Fetch all, then filter
            include_extra=False,  # Don't include extra data for efficiency
        )
        
        # Filter by task_id range if specified
        if task_id_start is not None or task_id_end is not None:
            filtered_samples = []
            for s in samples:
                task_id = s.get('task_id')
                if task_id is not None:
                    # Convert to int
                    if isinstance(task_id, str):
                        try:
                            task_id_int = int(task_id)
                        except ValueError:
                            continue
                    else:
                        task_id_int = int(task_id)
                    
                    # Check range
                    if task_id_start is not None and task_id_int < task_id_start:
                        continue
                    if task_id_end is not None and task_id_int >= task_id_end:
                        continue
                    
                    filtered_samples.append(s)
            samples = filtered_samples
        
        # Deduplicate by task_id if requested
        if deduplicate:
            task_id_samples = {}
            for sample in samples:
                task_id = sample.get('task_id')
                if task_id is not None:
                    # Convert to int for comparison
                    if isinstance(task_id, str):
                        try:
                            task_id_int = int(task_id)
                        except ValueError:
                            continue
                    else:
                        task_id_int = int(task_id)
                    
                    # Keep the latest (newest) sample
                    if task_id_int not in task_id_samples:
                        task_id_samples[task_id_int] = sample
                    else:
                        existing_ts = task_id_samples[task_id_int].get('timestamp', 0)
                        new_ts = sample.get('timestamp', 0)
                        if new_ts > existing_ts:
                            task_id_samples[task_id_int] = sample
            
            samples = list(task_id_samples.values())
        
        # Apply limit
        if len(samples) > limit:
            # Sort by timestamp descending first, then apply limit
            samples.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            samples = samples[:limit]
        
        # Convert to response models
        sample_details = [
            SampleDetail(
                timestamp=s["timestamp"],
                env=s["env"],
                task_id=s.get("task_id"),
                score=s["score"],
                latency_ms=s["latency_ms"],
                signature=s["signature"],
            )
            for s in samples
        ]
        
        # Create pagination info
        pagination = PaginationInfo(
            total=len(sample_details),
            limit=limit,
            next_cursor=None,
        )
        
        return SampleListResponse(
            samples=sample_details,
            pagination=pagination
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve samples: {str(e)}"
        )


@router.get("/uid/{uid}", response_model=SampleListResponse, dependencies=[Depends(rate_limit_read)])
async def get_uid_samples(
    uid: int,
    envs: str = Query("affine:sat,affine:abd,affine:ded", description="Comma-separated environments"),
    task_id_start: Optional[int] = Query(None, description="Task ID range start (inclusive)"),
    task_id_end: Optional[int] = Query(None, description="Task ID range end (exclusive)"),
    deduplicate: bool = Query(True, description="Deduplicate by task_id, keeping latest sample"),
    limit_per_env: int = Query(1000, description="Maximum results per environment", ge=1, le=10000),
    include_extra: bool = Query(False, description="Include extra field (conversation data, requires decompression)"),
    dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Get recent samples for a specific UID across multiple environments.
    
    This endpoint:
    1. Queries bittensor metagraph to get miner info (hotkey + model_revision)
    2. Queries samples across specified environments using hotkey+revision
    3. Returns combined results with optional filtering and deduplication
    
    If the UID is not currently assigned or has no valid commit, returns empty list.
    
    Query parameters:
    - envs: Comma-separated environment names (default: affine:sat,affine:abd,affine:ded)
    - task_id_start: Start of task ID range (inclusive)
    - task_id_end: End of task ID range (exclusive)
    - deduplicate: If True (default), keep only latest sample per task_id within each env
    - limit_per_env: Maximum results per environment (default: 1000, max: 10000)
    - include_extra: Include extra field with conversation data (default: False)
      Setting to False saves bandwidth and decompression cost
    
    Note: model_revision is queried from bittensor commits, not provided by user.
    """
    try:
        # Query bittensor metagraph to get miner info (including model_revision)
        # Reference: affine.miners.miners() implementation
        miners_dict = await get_miners(uids=uid, check_validity=False)
        
        if not miners_dict or uid not in miners_dict:
            # UID not currently assigned or no valid commit
            return SampleListResponse(
                samples=[],
                pagination=PaginationInfo(total=0, limit=limit_per_env, next_cursor=None)
            )
        
        miner = miners_dict[uid]
        hotkey = miner.hotkey
        model_revision = miner.revision
        
        if not model_revision:
            # No revision in commit, cannot query samples
            return SampleListResponse(
                samples=[],
                pagination=PaginationInfo(total=0, limit=limit_per_env, next_cursor=None)
            )
        
        # Parse environments
        env_list = [e.strip() for e in envs.split(",")]
        
        # Get samples across all environments
        results_by_env = await dao.get_samples_by_miner_all_envs(
            miner_hotkey=hotkey,
            model_revision=model_revision,
            envs=env_list,
            limit_per_env=None,  # Fetch all, then filter
            include_extra=include_extra,
        )
        
        # Process each environment's samples
        all_samples = []
        for env_name, env_samples in results_by_env.items():
            # Filter by task_id range if specified
            if task_id_start is not None or task_id_end is not None:
                filtered_samples = []
                for s in env_samples:
                    task_id = s.get('task_id')
                    if task_id is not None:
                        # Convert to int
                        if isinstance(task_id, str):
                            try:
                                task_id_int = int(task_id)
                            except ValueError:
                                continue
                        else:
                            task_id_int = int(task_id)
                        
                        # Check range
                        if task_id_start is not None and task_id_int < task_id_start:
                            continue
                        if task_id_end is not None and task_id_int >= task_id_end:
                            continue
                        
                        filtered_samples.append(s)
                env_samples = filtered_samples
            
            # Deduplicate by task_id within this env if requested
            if deduplicate:
                task_id_samples = {}
                for sample in env_samples:
                    task_id = sample.get('task_id')
                    if task_id is not None:
                        # Convert to int for comparison
                        if isinstance(task_id, str):
                            try:
                                task_id_int = int(task_id)
                            except ValueError:
                                continue
                        else:
                            task_id_int = int(task_id)
                        
                        # Keep the latest (newest) sample
                        if task_id_int not in task_id_samples:
                            task_id_samples[task_id_int] = sample
                        else:
                            existing_ts = task_id_samples[task_id_int].get('timestamp', 0)
                            new_ts = sample.get('timestamp', 0)
                            if new_ts > existing_ts:
                                task_id_samples[task_id_int] = sample
                
                env_samples = list(task_id_samples.values())
            
            # Apply limit per env
            if len(env_samples) > limit_per_env:
                # Sort by timestamp descending first, then apply limit
                env_samples.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                env_samples = env_samples[:limit_per_env]
            
            all_samples.extend(env_samples)
        
        # Convert to response models
        sample_details = [
            SampleDetail(
                timestamp=s["timestamp"],
                env=s["env"],
                task_id=s.get("task_id"),
                score=s["score"],
                latency_ms=s["latency_ms"],
                signature=s["signature"],
                extra=s.get("extra") if include_extra else None,
            )
            for s in all_samples
        ]
        
        # Sort by timestamp descending
        sample_details.sort(key=lambda x: x.timestamp, reverse=True)
        
        pagination = PaginationInfo(
            total=len(sample_details),
            limit=limit_per_env * len(env_list),
            next_cursor=None,
        )
        
        return SampleListResponse(
            samples=sample_details,
            pagination=pagination
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve samples: {str(e)}"
        )


@router.get("/timestamp-range", response_model=SampleListResponse, dependencies=[Depends(rate_limit_read)])
async def get_samples_by_timestamp(
    start_time: int = Query(..., description="Start timestamp in milliseconds"),
    end_time: int = Query(..., description="End timestamp in milliseconds"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Get samples by timestamp range.
    
    Uses timestamp-index GSI for efficient time-based queries.
    
    Query parameters:
    - start_time: Start timestamp in milliseconds (required)
    - end_time: End timestamp in milliseconds (required)
    - limit: Maximum results to return (default: 100, max: 1000)
    """
    try:
        # Query using timestamp GSI
        samples = await dao.get_samples_by_timestamp_range(
            start_timestamp=start_time,
            end_timestamp=end_time,
            limit=limit,
        )
        
        # Convert to response models
        sample_details = [
            SampleDetail(
                timestamp=s["timestamp"],
                env=s["env"],
                score=s["score"],
                latency_ms=s["latency_ms"],
                signature=s["signature"],
            )
            for s in samples
        ]
        
        pagination = PaginationInfo(
            total=len(samples),
            limit=limit,
            next_cursor=None,
        )
        
        return SampleListResponse(
            samples=sample_details,
            pagination=pagination
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve samples: {str(e)}"
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
    """
    try:
        # Construct PK and SK from natural keys
        pk = dao._make_pk(hotkey, model_revision, env)
        sk = dao._make_sk(task_id)
        
        # Get item directly using PK/SK
        from affine.database.client import get_client
        client = get_client()
        
        response = await client.get_item(
            TableName=dao.table_name,
            Key={
                'pk': {'S': pk},
                'sk': {'S': sk}
            }
        )
        
        if 'Item' not in response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample not found for hotkey={hotkey}, env={env}, task_id={task_id}"
            )
        
        # Deserialize and decompress
        item = dao._deserialize(response['Item'])
        
        if 'extra_compressed' in item:
            import json
            compressed = item['extra_compressed']
            extra_json = dao.decompress_data(compressed)
            extra = json.loads(extra_json)
        else:
            extra = item.get('extra', {})
        
        return SampleFullResponse(
            miner_hotkey=item["miner_hotkey"],
            model_revision=item["model_revision"],
            env=item["env"],
            score=item["score"],
            signature=item["signature"],
            extra=extra,
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