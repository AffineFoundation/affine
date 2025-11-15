"""
Admin Operations Router

Administrative endpoints for data retention, cleanup, and system management.
"""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from affine.api.models import (
    ProtectMinerRequest,
    ProtectMinerResponse,
    UnprotectMinerResponse,
    CleanupRequest,
    CleanupResponse,
    TaskGenerationRequest,
    TaskGenerationResponse,
    TaskCleanupResponse,
    ActiveMinersResponse,
    MinerInfo as MinerInfoModel,
)
from affine.api.dependencies import (
    get_data_retention_dao,
    get_sample_results_dao,
    get_task_generator_service,
    verify_admin_access,
    rate_limit_admin,
)
from affine.database.dao.data_retention import DataRetentionDAO
from affine.database.dao.sample_results import SampleResultsDAO
from affine.api.services.task_generator import TaskGeneratorService, MinerInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post(
    "/retention/protect",
    response_model=ProtectMinerResponse,
    dependencies=[Depends(rate_limit_admin)]
)
async def protect_miner(
    request: Request,
    data: ProtectMinerRequest,
    dao: DataRetentionDAO = Depends(get_data_retention_dao),
    admin_hotkey: str = Depends(verify_admin_access),
):
    """
    Protect a miner from data retention cleanup (requires admin auth).
    
    Protected miners (e.g., historical top-3 performers) will not have their
    samples automatically deleted, regardless of age.
    
    Request body:
    - miner_hotkey: Hotkey of the miner to protect
    - reason: Reason for protection (e.g., "Historical top-3 performer")
    """
    try:
        dao.set_protected(
            miner_hotkey=data.miner_hotkey,
            reason=data.reason,
        )
        
        return ProtectMinerResponse(
            miner_hotkey=data.miner_hotkey,
            protected=True,
            reason=data.reason,
            protected_at=int(time.time()),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to protect miner: {str(e)}"
        )


@router.delete(
    "/retention/protect/{hotkey}",
    response_model=UnprotectMinerResponse,
    dependencies=[Depends(rate_limit_admin)]
)
async def unprotect_miner(
    hotkey: str,
    request: Request,
    dao: DataRetentionDAO = Depends(get_data_retention_dao),
    admin_hotkey: str = Depends(verify_admin_access),
):
    """
    Remove protection from a miner (requires admin auth).
    
    The miner's samples will be subject to normal retention policies.
    """
    try:
        dao.remove_protected(hotkey)
        
        return UnprotectMinerResponse(
            miner_hotkey=hotkey,
            protected=False,
            message="Protection removed",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove protection: {str(e)}"
        )


@router.post(
    "/cleanup/samples",
    response_model=CleanupResponse,
    dependencies=[Depends(rate_limit_admin)]
)
async def cleanup_samples(
    request: Request,
    data: CleanupRequest,
    retention_dao: DataRetentionDAO = Depends(get_data_retention_dao),
    samples_dao: SampleResultsDAO = Depends(get_sample_results_dao),
    admin_hotkey: str = Depends(verify_admin_access),
):
    """
    Trigger sample cleanup for non-protected miners (requires admin auth).
    
    Deletes samples older than the specified retention period, except for
    protected miners.
    
    Request body:
    - retention_days: Keep samples newer than this many days (default: 90)
    - dry_run: If true, simulate cleanup without deleting (default: false)
    """
    try:
        # Get protected miners
        protected_miners = retention_dao.get_protected_miners()
        protected_hotkeys = [m["miner_hotkey"] for m in protected_miners]
        
        # Calculate cutoff timestamp
        cutoff_timestamp = int(time.time()) - (data.retention_days * 86400)
        
        deleted_count = 0
        
        if not data.dry_run:
            # Delete samples for non-protected miners
            # Note: This is a simplified implementation
            # In production, you'd want to batch this operation
            deleted_count = samples_dao.delete_samples_before(
                timestamp=cutoff_timestamp,
                exclude_hotkeys=protected_hotkeys,
            )
        else:
            # Dry run - just count what would be deleted
            # This would require additional DAO methods to count without deleting
            deleted_count = 0  # Placeholder
        
        return CleanupResponse(
            deleted_count=deleted_count,
            protected_count=len(protected_hotkeys),
            dry_run=data.dry_run,
            message=f"Cleanup {'simulation' if data.dry_run else 'completed'}: {deleted_count} samples {'would be' if data.dry_run else ''} deleted",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup samples: {str(e)}"
        )


@router.post(
    "/tasks/generate",
    response_model=TaskGenerationResponse,
    dependencies=[Depends(rate_limit_admin)]
)
async def generate_tasks(
    request: Request,
    data: TaskGenerationRequest,
    task_generator: TaskGeneratorService = Depends(get_task_generator_service),
    admin_hotkey: str = Depends(verify_admin_access),
):
    """
    Manually trigger task generation for all active miners (requires admin auth).
    
    This endpoint:
    1. Fetches all active miners from bittensor/chutes
    2. For each miner and environment, generates missing task_ids
    3. Returns a report of created tasks
    
    Request body:
    - envs: List of environments to generate tasks for (default: all)
    - max_tasks_per_miner_env: Maximum tasks per miner/env (default: 100)
    """
    try:
        logger.info(f"Task generation triggered by admin {admin_hotkey[:8]}...")
        
        # Fetch active miners from bittensor
        from affine.miners import miners as fetch_miners
        
        miner_dict = await fetch_miners()
        
        # Convert to MinerInfo format
        miners = [
            MinerInfo(
                hotkey=m.hotkey,
                model_revision=m.revision,  # Miner model uses 'revision', MinerInfo uses 'model_revision'
                model=m.model,
                uid=m.uid
            )
            for m in miner_dict.values()
        ]
        
        logger.info(f"Found {len(miners)} active miners")
        
        # Generate tasks
        result = await task_generator.generate_all_tasks(
            miners=miners,
            envs=data.envs if data.envs else None,  # None means all envs
            max_tasks_per_miner_env=data.max_tasks_per_miner_env
        )
        
        return TaskGenerationResponse(
            total_tasks_created=result.total_tasks_created,
            tasks_by_env=result.tasks_by_env,
            miners_processed=result.miners_processed,
            errors=result.errors,
            timestamp=int(time.time())
        )
        
    except Exception as e:
        logger.error(f"Task generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate tasks: {str(e)}"
        )


@router.post(
    "/tasks/cleanup",
    response_model=TaskCleanupResponse,
    dependencies=[Depends(rate_limit_admin)]
)
async def cleanup_invalid_tasks(
    request: Request,
    task_generator: TaskGeneratorService = Depends(get_task_generator_service),
    admin_hotkey: str = Depends(verify_admin_access),
):
    """
    Clean up invalid tasks from the queue (requires admin auth).
    
    Removes tasks for miners that are no longer active (hotkey not in metagraph
    or model_revision mismatch).
    """
    try:
        logger.info(f"Task cleanup triggered by admin {admin_hotkey[:8]}...")
        
        # Fetch active miners
        from affine.miners import miners as fetch_miners
        
        miner_dict = await fetch_miners()
        
        # Convert to MinerInfo format
        valid_miners = [
            MinerInfo(
                hotkey=m.hotkey,
                model_revision=m.revision,  # Miner model uses 'revision', MinerInfo uses 'model_revision'
                model=m.model,
                uid=m.uid
            )
            for m in miner_dict.values()
        ]
        
        # Clean up invalid tasks
        removed_count = await task_generator.cleanup_invalid_tasks(valid_miners)
        
        logger.info(f"Cleaned up {removed_count} invalid tasks")
        
        return TaskCleanupResponse(
            removed_count=removed_count,
            timestamp=int(time.time())
        )
        
    except Exception as e:
        logger.error(f"Task cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup tasks: {str(e)}"
        )


@router.get(
    "/miners/active",
    response_model=ActiveMinersResponse,
    dependencies=[Depends(rate_limit_admin)]
)
async def get_active_miners(
    request: Request,
    admin_hotkey: str = Depends(verify_admin_access),
):
    """
    Get list of all active miners from bittensor (requires admin auth).
    
    Returns miners that:
    1. Are registered in bittensor metagraph
    2. Have a valid chutes deployment
    3. Model revision matches
    """
    try:
        from affine.miners import miners as fetch_miners
        
        miner_dict = await fetch_miners()
        
        miners = [
            MinerInfoModel(
                hotkey=m.hotkey,
                model_revision=m.revision,  # Miner model uses 'revision', MinerInfo uses 'model_revision'
                model=m.model,
                uid=m.uid
            )
            for m in miner_dict.values()
        ]
        
        return ActiveMinersResponse(
            miners=miners,
            total=len(miners)
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch active miners: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch active miners: {str(e)}"
        )