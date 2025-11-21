"""
Sample Results Router

Endpoints for submitting and querying sample results.
"""

import time
import uuid
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
    get_task_pool_manager,
    verify_executor_auth,
    rate_limit_read,
    rate_limit_write,
)
from affine.api.config import config
from affine.core.models import SampleSubmission
from affine.api.utils.pagination import get_pagination_params, create_pagination_info
from affine.database.dao.sample_results import SampleResultsDAO
from affine.api.services.task_pool import TaskPoolManager
from affine.api.utils.bittensor import get_subtensor

router = APIRouter(prefix="/samples", tags=["Samples"])


@router.post("/submit", response_model=SampleSubmitResponse, dependencies=[Depends(rate_limit_write)])
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
        # Query from miners monitor (fast, no slow queries)
        from affine.affine.backend.miners.miners_monitor import MinersMonitor
        
        miners_monitor = MinersMonitor.get_instance()
        miners_dict = await miners_monitor.get_valid_miners()
        
        # Find miner by uid
        miner_info = None
        for info in miners_dict.values():
            if info.uid == uid:
                miner_info = info
                break
        
        if not miner_info:
            # UID not found in cache
            return SampleListResponse(
                samples=[],
                pagination=PaginationInfo(total=0, limit=limit_per_env, next_cursor=None)
            )
        
        hotkey = miner_info.hotkey
        model_revision = miner_info.revision
        
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


# Cache for scoring endpoint (30-minute TTL)
_scoring_cache: Dict[str, Any] = {}
_scoring_cache_time: int = 0
SCORING_CACHE_TTL = 1800  # 30 minutes in seconds


@router.get("/scoring", dependencies=[Depends(rate_limit_read)])
async def get_scoring_data(
    dao: SampleResultsDAO = Depends(get_sample_results_dao),
):
    """
    Get compressed and deduplicated scoring data for all valid miners.
    
    This endpoint:
    1. Queries system_config for scoring environments and ranges
    2. Gets all valid miners from miners table
    3. Fetches and deduplicates samples for each miner in each environment
    4. Returns compressed data structure for score calculation
    
    Response format matches design document specification.
    
    Cache: 30-minute TTL to reduce load on repeated score calculations.
    """
    global _scoring_cache, _scoring_cache_time
    
    # Check cache validity
    current_time = int(time.time())
    if _scoring_cache and (current_time - _scoring_cache_time) < SCORING_CACHE_TTL:
        return _scoring_cache
    
    try:
        # Import dependencies
        from affine.database.dao.system_config import SystemConfigDAO
        from affine.database.dao.miners import MinersDAO
        
        system_config_dao = SystemConfigDAO()
        miners_dao = MinersDAO()
        
        # Get scoring configuration
        scoring_envs = await system_config_dao.get_active_environments()
        if not scoring_envs:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No scoring environments configured"
            )
        
        env_ranges = await system_config_dao.get_env_task_ranges()
        
        # Get all valid miners
        valid_miners = await miners_dao.get_valid_miners()
        if not valid_miners:
            # Return empty result if no valid miners
            return {}
        
        # Build result structure
        result = {}
        
        for miner in valid_miners:
            uid = miner.get('uid')
            hotkey = miner.get('hotkey')
            model_revision = miner.get('revision')
            
            if not all([uid is not None, hotkey, model_revision]):
                continue
            
            # Initialize miner entry
            miner_entry = {
                'hotkey': hotkey,
                'model_revision': model_revision,
                'env': {}
            }
            
            # Fetch samples for each environment
            for env_name in scoring_envs:
                # Get scoring range for this environment
                scoring_range = env_ranges.get(env_name, {}).get('scoring_range', [0, 0])
                start_id, end_id = scoring_range
                
                if start_id >= end_id:
                    continue
                
                # Fetch samples for this env
                samples = await dao.get_samples_by_miner(
                    miner_hotkey=hotkey,
                    model_revision=model_revision,
                    env=env_name,
                    limit=None,
                    include_extra=False,  # Don't include extra for scoring
                )
                
                # Filter by scoring range and deduplicate
                task_id_samples = {}
                for sample in samples:
                    task_id_str = sample.get('task_id')
                    if not task_id_str:
                        continue
                    
                    # Convert to int
                    try:
                        task_id_int = int(task_id_str)
                    except (ValueError, TypeError):
                        continue
                    
                    # Check if in scoring range
                    if task_id_int < start_id or task_id_int >= end_id:
                        continue
                    
                    # Deduplicate: keep latest sample per task_id
                    if task_id_int not in task_id_samples:
                        task_id_samples[task_id_int] = sample
                    else:
                        existing_ts = task_id_samples[task_id_int].get('timestamp', 0)
                        new_ts = sample.get('timestamp', 0)
                        if new_ts > existing_ts:
                            task_id_samples[task_id_int] = sample
                
                # Build samples list for this env
                env_samples = []
                for task_id_int in sorted(task_id_samples.keys()):
                    sample = task_id_samples[task_id_int]
                    env_samples.append({
                        'task_id': task_id_int,
                        'score': sample.get('score', 0.0),
                        'task_uuid': sample.get('timestamp'),  # Use timestamp as task_uuid
                        'timestamp': sample.get('timestamp'),
                    })
                
                # Calculate completeness
                expected_count = end_id - start_id
                completed_count = len(env_samples)
                completeness = completed_count / expected_count if expected_count > 0 else 0.0
                
                # Find missing task IDs
                completed_task_ids = set(task_id_samples.keys())
                all_task_ids = set(range(start_id, end_id))
                missing_task_ids = sorted(list(all_task_ids - completed_task_ids))
                
                # Add environment data
                miner_entry['env'][env_name] = {
                    'samples': env_samples,
                    'total_count': expected_count,
                    'completed_count': completed_count,
                    'missing_task_ids': missing_task_ids[:100],  # Limit to first 100
                    'completeness': round(completeness, 4)
                }
            
            # Add miner to result
            result[str(uid)] = miner_entry
        
        # Update cache
        _scoring_cache = result
        _scoring_cache_time = current_time
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate scoring data: {str(e)}"
        )