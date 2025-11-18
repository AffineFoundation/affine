"""
Configuration Management Router

Provides REST API endpoints for dynamic configuration management.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional, List, Dict
from affine.database.dao.system_config import SystemConfigDAO

router = APIRouter(prefix="/api/v1/config", tags=["config"])
config_dao = SystemConfigDAO()


class ConfigSetRequest(BaseModel):
    """Request model for setting configuration."""
    param_value: Any
    param_type: str  # str/int/float/bool/dict/list
    description: str = ""
    updated_by: str = "api"


class ConfigBatchRequest(BaseModel):
    """Request model for batch config retrieval."""
    keys: List[str]


class ConfigRefreshRequest(BaseModel):
    """Request model for refreshing config cache."""
    keys: Optional[List[str]] = None


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


@router.put("/{key}")
async def set_config(key: str, request: ConfigSetRequest):
    """Set or update a configuration parameter.
    
    Args:
        key: Configuration key
        request: Config value and metadata
        
    Returns:
        Updated config item
        
    Example:
        PUT /api/v1/config/scheduler.check_interval
        Body: {
            "param_value": 15,
            "param_type": "int",
            "description": "Adjusted to 15 seconds",
            "updated_by": "admin"
        }
    """
    result = await config_dao.set_param(
        param_name=key,
        param_value=request.param_value,
        param_type=request.param_type,
        description=request.description,
        updated_by=request.updated_by
    )
    
    return {"message": f"Config '{key}' updated", "config": result}


@router.delete("/{key}")
async def delete_config(key: str):
    """Delete a configuration parameter (revert to default).
    
    Args:
        key: Configuration key
        
    Returns:
        Success message
        
    Raises:
        404: Config not found
        
    Example:
        DELETE /api/v1/config/scheduler.check_interval
        Will revert to code default value
    """
    success = await config_dao.delete_param(key)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Config '{key}' not found")
    
    return {"message": f"Config '{key}' deleted (will use default value)"}


@router.post("/batch")
async def get_configs_batch(request: ConfigBatchRequest):
    """Batch retrieve multiple configuration parameters.
    
    Args:
        request: List of config keys
        
    Returns:
        Dictionary mapping keys to values
        
    Example:
        POST /api/v1/config/batch
        Body: {
            "keys": [
                "scheduler.check_interval",
                "executor.workers_per_env"
            ]
        }
    """
    configs = {}
    for key in request.keys:
        value = await config_dao.get_param_value(key)
        configs[key] = value
    
    return {"configs": configs}


@router.post("/refresh")
async def refresh_configs(request: Optional[ConfigRefreshRequest] = None):
    """Refresh configuration cache (notification endpoint).
    
    This endpoint serves as a notification mechanism. Actual cache refresh
    is handled by each service's DynamicConfig instance based on TTL.
    
    Args:
        request: Optional list of specific keys to refresh
        
    Returns:
        Confirmation message
        
    Example:
        POST /api/v1/config/refresh
        Body: {"keys": ["scheduler.check_interval"]}
    """
    keys = request.keys if request else None
    
    return {
        "message": "Config refresh notification sent",
        "keys": keys or "all",
        "note": "Services will refresh based on cache TTL (60s)"
    }


@router.get("/list/all")
async def list_all_configs_detailed():
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


@router.get("/dataset/{env}")
async def get_dataset_info(env: str):
    """Get dataset information for a specific environment.
    
    Args:
        env: Environment name (e.g., affine:sat, affine:abd)
        
    Returns:
        Dataset configuration including length and task_id range
        
    Example:
        GET /api/v1/config/dataset/affine:sat
    """
    # Default dataset lengths by environment
    # These can be overridden in system_config table
    default_lengths = {
        "affine:sat": 100,
        "affine:abd": 100,
        "affine:ded": 100,
        "agentgym:webshop": 50,
    }
    
    # Try to get from config first
    config_key = f"dataset.{env}.length"
    dataset_length = await config_dao.get_param_value(config_key)
    
    if dataset_length is None:
        # Use default
        dataset_length = default_lengths.get(env, 100)
    
    # Check for task_id range override (for dataset expansion transitions)
    task_id_start_key = f"dataset.{env}.task_id_start"
    task_id_end_key = f"dataset.{env}.task_id_end"
    
    task_id_start = await config_dao.get_param_value(task_id_start_key)
    task_id_end = await config_dao.get_param_value(task_id_end_key)
    
    # Default range is 0 to dataset_length
    if task_id_start is None:
        task_id_start = 0
    if task_id_end is None:
        task_id_end = dataset_length
    
    return {
        "env": env,
        "dataset_length": dataset_length,
        "task_id_start": task_id_start,
        "task_id_end": task_id_end,
        "note": "task_id_start/end can be overridden for dataset expansion transitions"
    }


# Environment Configuration Management


class EnvironmentListRequest(BaseModel):
    """Request model for environment list update."""
    environments: List[str]
    updated_by: str = "api"


class EnvTaskRangesRequest(BaseModel):
    """Request model for environment task ranges update."""
    ranges: Dict[str, Dict[str, Any]]
    updated_by: str = "api"


@router.get("/sampling-environments")
async def get_sampling_environments():
    """Get list of environments for task generation (sampling).
    
    Returns:
        List of environment names for sampling
        
    Example:
        GET /api/v1/config/sampling-environments
        Response: {"environments": ["affine:sat", "affine:abd"]}
    """
    envs = await config_dao.get_sampling_environments()
    return {"environments": envs}


@router.put("/sampling-environments")
async def set_sampling_environments(request: EnvironmentListRequest):
    """Set list of environments for task generation (sampling).
    
    Args:
        request: List of environment names and updated_by
        
    Returns:
        Updated config item
        
    Example:
        PUT /api/v1/config/sampling-environments
        Body: {
            "environments": ["affine:sat", "affine:abd", "affine:ded"],
            "updated_by": "admin"
        }
    """
    result = await config_dao.set_sampling_environments(
        envs=request.environments,
        updated_by=request.updated_by
    )
    return {"message": "Sampling environments updated", "config": result}


@router.get("/active-environments")
async def get_active_environments():
    """Get list of environments for score calculation (active).
    
    Allows transition period when adding new environments:
    - New env added to sampling list first
    - After sufficient sampling, added to active list
    
    Returns:
        List of environment names for scoring
        
    Example:
        GET /api/v1/config/active-environments
        Response: {"environments": ["affine:sat", "affine:abd"]}
    """
    envs = await config_dao.get_active_environments()
    return {"environments": envs}


@router.put("/active-environments")
async def set_active_environments(request: EnvironmentListRequest):
    """Set list of environments for score calculation (active).
    
    Args:
        request: List of environment names and updated_by
        
    Returns:
        Updated config item
        
    Example:
        PUT /api/v1/config/active-environments
        Body: {
            "environments": ["affine:sat", "affine:abd"],
            "updated_by": "admin"
        }
    """
    result = await config_dao.set_active_environments(
        envs=request.environments,
        updated_by=request.updated_by
    )
    return {"message": "Active environments updated", "config": result}


@router.get("/env-task-ranges")
async def get_env_task_ranges():
    """Get task_id ranges for each environment.
    
    Returns:
        Dictionary mapping env names to range configurations
        
    Example:
        GET /api/v1/config/env-task-ranges
        Response: {
            "ranges": {
                "affine:sat": {
                    "sampling_range": [0, 1000],
                    "active_range": [0, 1000]
                },
                "affine:abd": {
                    "sampling_range": [0, 500],
                    "active_range": [0, 300]
                }
            }
        }
    """
    ranges = await config_dao.get_env_task_ranges()
    return {"ranges": ranges}


@router.put("/env-task-ranges")
async def set_env_task_ranges(request: EnvTaskRangesRequest):
    """Set task_id ranges for environments.
    
    Args:
        request: Dictionary mapping env names to range configs
        
    Returns:
        Updated config item
        
    Example:
        PUT /api/v1/config/env-task-ranges
        Body: {
            "ranges": {
                "affine:sat": {
                    "sampling_range": [0, 1000],
                    "active_range": [0, 1000]
                }
            },
            "updated_by": "admin"
        }
    """
    result = await config_dao.set_env_task_ranges(
        ranges=request.ranges,
        updated_by=request.updated_by
    )
    return {"message": "Environment task ranges updated", "config": result}


@router.get("/env/{env}/sampling-range")
async def get_env_sampling_range(env: str):
    """Get sampling range for a specific environment.
    
    Args:
        env: Environment name
        
    Returns:
        Sampling range (start_id, end_id)
        
    Example:
        GET /api/v1/config/env/affine:sat/sampling-range
        Response: {"env": "affine:sat", "start_id": 0, "end_id": 1000}
    """
    start_id, end_id = await config_dao.get_env_sampling_range(env)
    return {
        "env": env,
        "start_id": start_id,
        "end_id": end_id
    }


@router.get("/env/{env}/active-range")
async def get_env_active_range(env: str):
    """Get active range for a specific environment.
    
    Args:
        env: Environment name
        
    Returns:
        Active range (start_id, end_id)
        
    Example:
        GET /api/v1/config/env/affine:sat/active-range
        Response: {"env": "affine:sat", "start_id": 0, "end_id": 1000}
    """
    start_id, end_id = await config_dao.get_env_active_range(env)
    return {
        "env": env,
        "start_id": start_id,
        "end_id": end_id
    }