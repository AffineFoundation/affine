"""
Configuration Management Router

Provides REST API endpoints for dynamic configuration management.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from affine.database.dao.system_config import SystemConfigDAO

router = APIRouter(prefix="/api/v1/config", tags=["config"])
config_dao = SystemConfigDAO()


@router.get("")
async def get_all_configs(prefix: Optional[str] = None):
    """Get all configurations, optionally filtered by prefix.
    
    Args:
        prefix: Config key prefix filter (e.g., "scheduler.")
        
    Returns:
        Dictionary of all matching configs

    Example:
        GET /api/v1/config?prefix=scheduler
        Returns all scheduler.* configs
    """
    all_configs = await config_dao.get_all_params()
    
    if prefix:
        filtered = {k: v for k, v in all_configs.items() if k.startswith(prefix)}
        return {"configs": filtered}
    
    return {"configs": all_configs}


@router.get("/{key}")
async def get_config(key: str):
    """Get a single configuration parameter.
    
    Args:
        key: Configuration key
        
    Returns:
        Full config item with metadata
        
    Raises:
        404: Config not found
        
    Example:
        GET /api/v1/config/scheduler.check_interval
    """
    config = await config_dao.get_param(key)
    
    if not config:
        raise HTTPException(status_code=404, detail=f"Config '{key}' not found")
    
    return config

    """List all configuration parameters with full metadata.
    
    Returns:
        List of config items with complete details including
        version, updated_at, updated_by, etc.
        
    Example:
        GET /api/v1/config/list/all
    """
    configs = await config_dao.list_all_configs()
    
    return {
        "count": len(configs),
        "configs": configs
    }