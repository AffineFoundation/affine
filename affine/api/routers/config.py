"""
System Configuration Router

Endpoints for managing dynamic system configuration.
"""

import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
from affine.api.models import (
    ConfigListResponse,
    ConfigParameter,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
)
from affine.api.dependencies import (
    get_system_config_dao,
    verify_admin_access,
    rate_limit_read,
    rate_limit_admin,
)
from affine.database.dao.system_config import SystemConfigDAO

router = APIRouter(prefix="/config", tags=["Configuration"])


@router.get("", response_model=ConfigListResponse, dependencies=[Depends(rate_limit_read)])
async def get_all_config(
    dao: SystemConfigDAO = Depends(get_system_config_dao),
):
    """
    Get all configuration parameters.
    
    Returns a list of all system configuration parameters with their current values.
    """
    try:
        params = await dao.list_all_configs()
        
        config_params = [
            ConfigParameter(
                name=p["param_name"],
                value=p["param_value"],
                description=p.get("description", ""),
                version=p["version"],
                updated_at=p["updated_at"],
            )
            for p in params
        ]
        
        return ConfigListResponse(parameters=config_params)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )


@router.get("/{param_name}", response_model=ConfigParameter, dependencies=[Depends(rate_limit_read)])
async def get_config_param(
    param_name: str,
    dao: SystemConfigDAO = Depends(get_system_config_dao),
):
    """
    Get a specific configuration parameter.
    
    Returns the current value and metadata for the specified parameter.
    """
    try:
        param = await dao.get_param(param_name)
        
        if not param:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parameter '{param_name}' not found"
            )
        
        return ConfigParameter(
            name=param["param_name"],
            value=param["param_value"],
            description=param.get("description", ""),
            version=param["version"],
            updated_at=param["updated_at"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve parameter: {str(e)}"
        )


@router.put(
    "/{param_name}",
    response_model=ConfigUpdateResponse,
    dependencies=[Depends(rate_limit_admin)]
)
async def update_config_param(
    param_name: str,
    request: Request,
    data: ConfigUpdateRequest,
    dao: SystemConfigDAO = Depends(get_system_config_dao),
    admin_hotkey: str = Depends(verify_admin_access),
):
    """
    Update a configuration parameter (requires admin auth).
    
    Only administrators can modify system configuration.
    
    Request body:
    - value: New value for the parameter (any JSON type)
    - description: Description of the parameter
    """
    try:
        # Get current version (if exists)
        current = await dao.get_param(param_name)
        new_version = (current["version"] + 1) if current else 1
        
        # Determine parameter type
        param_type = type(data.value).__name__
        if isinstance(data.value, dict):
            param_type = "dict"
        elif isinstance(data.value, list):
            param_type = "list"
        
        # Update parameter
        await dao.set_param(
            param_name=param_name,
            param_value=data.value,
            param_type=param_type,
            description=data.description,
            updated_by=admin_hotkey,
        )
        
        return ConfigUpdateResponse(
            name=param_name,
            value=data.value,
            version=new_version,
            updated_at=int(time.time()),
            message="Parameter updated successfully",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update parameter: {str(e)}"
        )