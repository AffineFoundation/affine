"""
FastAPI Dependencies

Reusable dependencies for authentication, database access, etc.
"""

import time
from typing import Optional
from fastapi import Depends, HTTPException, Request, Header, status
from affine.api.config import config
from affine.api.utils.auth import get_hotkey_from_request, verify_request_signature
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.task_queue import TaskQueueDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.data_retention import DataRetentionDAO
from affine.api.services.task_generator import TaskGeneratorService
from affine.api.services.auth import AuthService


# Database DAOs (singleton instances)
_sample_results_dao: Optional[SampleResultsDAO] = None
_task_queue_dao: Optional[TaskQueueDAO] = None
_execution_logs_dao: Optional[ExecutionLogsDAO] = None
_scores_dao: Optional[ScoresDAO] = None
_system_config_dao: Optional[SystemConfigDAO] = None
_data_retention_dao: Optional[DataRetentionDAO] = None
_task_generator_service: Optional[TaskGeneratorService] = None
_auth_service: Optional[AuthService] = None


def get_sample_results_dao() -> SampleResultsDAO:
    """Get SampleResultsDAO instance."""
    global _sample_results_dao
    if _sample_results_dao is None:
        _sample_results_dao = SampleResultsDAO()
    return _sample_results_dao


def get_task_queue_dao() -> TaskQueueDAO:
    """Get TaskQueueDAO instance."""
    global _task_queue_dao
    if _task_queue_dao is None:
        _task_queue_dao = TaskQueueDAO()
    return _task_queue_dao


def get_execution_logs_dao() -> ExecutionLogsDAO:
    """Get ExecutionLogsDAO instance."""
    global _execution_logs_dao
    if _execution_logs_dao is None:
        _execution_logs_dao = ExecutionLogsDAO()
    return _execution_logs_dao


def get_scores_dao() -> ScoresDAO:
    """Get ScoresDAO instance."""
    global _scores_dao
    if _scores_dao is None:
        _scores_dao = ScoresDAO()
    return _scores_dao


def get_system_config_dao() -> SystemConfigDAO:
    """Get SystemConfigDAO instance."""
    global _system_config_dao
    if _system_config_dao is None:
        _system_config_dao = SystemConfigDAO()
    return _system_config_dao



def get_data_retention_dao() -> DataRetentionDAO:
    """Get DataRetentionDAO instance."""
    global _data_retention_dao
    if _data_retention_dao is None:
        _data_retention_dao = DataRetentionDAO()
    return _data_retention_dao


def get_task_generator_service() -> TaskGeneratorService:
    """Get TaskGeneratorService instance."""
    global _task_generator_service
    if _task_generator_service is None:
        _task_generator_service = TaskGeneratorService(
            sample_results_dao=get_sample_results_dao(),
            task_queue_dao=get_task_queue_dao()
        )
    return _task_generator_service


def get_auth_service() -> AuthService:
    """Get AuthService instance for executor authentication."""
    global _auth_service
    if _auth_service is None:
        # Create with non-strict mode for development
        # In production, use create_auth_service_from_chain()
        _auth_service = AuthService(
            authorized_validators=set(),
            signature_expiry_seconds=60,  # 1 minute timeout
            strict_mode=False  # Non-strict for development
        )
    return _auth_service


async def verify_executor_auth(
    executor_hotkey: str = Header(..., alias="X-Hotkey"),
    executor_signature: str = Header(..., alias="X-Signature"),
    executor_message: str = Header(..., alias="X-Message"),
    auth_service: AuthService = Depends(get_auth_service),
) -> str:
    """
    Verify executor authentication with timestamp-based message.
    
    This dependency validates:
    1. Message format is a valid timestamp (integer string)
    2. Timestamp is within 60 seconds (prevents replay attacks)
    3. Signature is valid for the message
    
    Args:
        executor_hotkey: Executor's hotkey from header
        executor_signature: Signature from header
        executor_message: Timestamp string from header
        auth_service: Auth service instance
    
    Returns:
        Validated executor hotkey
        
    Raises:
        HTTPException: If validation fails
    """
    # Validate message format (should be timestamp)
    try:
        timestamp = int(executor_message)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid message format: expected timestamp"
        )
    
    # Check timestamp is within 60 seconds
    current_time = int(time.time())
    time_diff = abs(current_time - timestamp)
    
    if time_diff > 60:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Message expired (timestamp diff: {time_diff}s, max: 60s)"
        )
    
    # Verify signature
    is_valid = auth_service.verify_signature(
        hotkey=executor_hotkey,
        message=executor_message,
        signature=executor_signature
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid executor signature"
        )
    
    # Check if authorized validator (optional in non-strict mode)
    if not auth_service.is_authorized_validator(executor_hotkey):
        if auth_service.strict_mode:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Executor not authorized"
            )
    
    return executor_hotkey


async def verify_signature_dependency(request: Request) -> str:
    """
    Dependency to verify request signature.
    
    Returns:
        Authenticated hotkey
        
    Raises:
        HTTPException: If signature is invalid
    """
    try:
        # Get body as dict
        body = await request.json()
        hotkey, is_valid = await verify_request_signature(request, body)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )
        
        return hotkey
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request body: {str(e)}"
        )


async def verify_admin_access(request: Request) -> str:
    """
    Dependency to verify admin access.
    
    Returns:
        Authenticated admin hotkey
        
    Raises:
        HTTPException: If not authorized as admin
    """
    hotkey = await verify_signature_dependency(request)
    
    if not config.is_admin(hotkey):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return hotkey


# Rate limiting storage (in-memory for now, could use Redis in production)
_rate_limit_store: dict = {}


def check_rate_limit(
    identifier: str,
    limit: int,
    window_seconds: int = 3600,
) -> bool:
    """
    Check if rate limit is exceeded.
    
    Args:
        identifier: Unique identifier (IP, hotkey, etc.)
        limit: Max requests per window
        window_seconds: Time window in seconds
        
    Returns:
        True if within limit, False if exceeded
    """
    current_time = int(time.time())
    window_start = current_time - window_seconds
    
    # Get or create request history for this identifier
    if identifier not in _rate_limit_store:
        _rate_limit_store[identifier] = []
    
    # Remove old requests outside the window
    _rate_limit_store[identifier] = [
        ts for ts in _rate_limit_store[identifier] if ts > window_start
    ]
    
    # Check if limit is exceeded
    if len(_rate_limit_store[identifier]) >= limit:
        return False
    
    # Add current request
    _rate_limit_store[identifier].append(current_time)
    return True


async def rate_limit_read(request: Request):
    """Dependency for read endpoint rate limiting."""
    if not config.RATE_LIMIT_ENABLED:
        return
    
    identifier = request.client.host
    if not check_rate_limit(identifier, config.RATE_LIMIT_READ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


async def rate_limit_write(request: Request):
    """Dependency for write endpoint rate limiting."""
    if not config.RATE_LIMIT_ENABLED:
        return
    
    # Use hotkey as identifier for write operations
    hotkey = get_hotkey_from_request(request)
    if not check_rate_limit(hotkey, config.RATE_LIMIT_WRITE):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


async def rate_limit_admin(request: Request):
    """Dependency for admin endpoint rate limiting."""
    if not config.RATE_LIMIT_ENABLED:
        return
    
    hotkey = get_hotkey_from_request(request)
    if not check_rate_limit(hotkey, config.RATE_LIMIT_ADMIN):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


async def get_system_config():
    """Get system configuration."""
    dao = get_system_config_dao()
    return await dao.get_config()