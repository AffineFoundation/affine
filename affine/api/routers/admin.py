"""
Admin Operations Router

Administrative endpoints for data retention, cleanup, and system management.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, Request, status
from affine.api.models import (
    ProtectMinerRequest,
    ProtectMinerResponse,
    UnprotectMinerResponse,
    CleanupRequest,
    CleanupResponse,
)
from affine.api.dependencies import (
    get_data_retention_dao,
    get_sample_results_dao,
    verify_admin_access,
    rate_limit_admin,
)
from affine.database.dao.data_retention import DataRetentionDAO
from affine.database.dao.sample_results import SampleResultsDAO

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