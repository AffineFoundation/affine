"""
Health Check Router

Endpoints for health checks and metrics.
"""

import time
from fastapi import APIRouter, Depends
from affine.api.models import HealthResponse
from affine.api.config import config
from affine.api.dependencies import rate_limit_read
from affine.database.client import get_client

router = APIRouter(prefix="", tags=["Health"])

# Track server start time
_server_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API service health.
    
    Returns service status, version, and database connectivity.
    """
    # Check database connection
    try:
        client = get_client()
        # Simple connectivity test
        await client.list_tables(Limit=1)
        database_status = "connected"
    except Exception as e:
        database_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy" if database_status == "connected" else "degraded",
        timestamp=int(time.time()),
        version=config.APP_VERSION,
        database=database_status,
        uptime_seconds=int(time.time() - _server_start_time),
    )


@router.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    # In a production system, you would use prometheus_client library
    # For now, return a simple text response
    uptime = int(time.time() - _server_start_time)
    
    metrics_text = f"""# HELP api_uptime_seconds API server uptime in seconds
# TYPE api_uptime_seconds gauge
api_uptime_seconds {uptime}

# HELP api_info API server information
# TYPE api_info gauge
api_info{{version="{config.APP_VERSION}"}} 1
"""
    
    return metrics_text