"""
Affine API Server

FastAPI application entry point.
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from affine.api.config import config
from affine.api.middleware import setup_middleware
from affine.api.routers import (
    samples_router,
    tasks_router,
    miners_router,
    scores_router,
    config_router,
    logs_router,
)
from affine.database import init_client, close_client

# Configure logging using setup_logging
from affine.core.setup import setup_logging, logger

# Map LOG_LEVEL string to verbosity
log_level = os.getenv("API_LOG_LEVEL", config.LOG_LEVEL).upper()
verbosity_map = {
    "CRITICAL": 0,
    "ERROR": 0,
    "WARNING": 0,
    "INFO": 1,
    "DEBUG": 2,
    "TRACE": 3,
}
verbosity = verbosity_map.get(log_level, 1)
setup_logging(verbosity)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Affine API server...")
    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database client: {e}")
        raise
    
    # Initialize TaskPoolManager (required for /fetch and /submit endpoints)
    task_pool = None
    
    if config.SERVICES_ENABLED:
        logger.info("Services mode enabled (SERVICES_ENABLED=true)")
        
        try:
            from affine.api.services.task_pool import TaskPoolManager
            
            # Initialize TaskPoolManager
            task_pool = await TaskPoolManager.initialize(
                lock_timeout_seconds=300,  # 5 minutes
                cleanup_interval_seconds=60  # 1 minute
            )
            logger.info("TaskPoolManager initialized with background tasks")
            
        except Exception as e:
            logger.error(f"Failed to initialize TaskPoolManager: {e}")
            raise
    else:
        logger.warning(
            "Services mode disabled (SERVICES_ENABLED=false). "
            "TaskPoolManager will not start. "
            "/fetch and /submit endpoints will return 503 Service Unavailable."
        )
    
    # Initialize scoring cache
    scoring_refresh_task = None
    try:
        from affine.api.services.scoring_cache import warmup_cache, refresh_cache_loop
        
        # Warmup cache on startup
        await warmup_cache()
        
        # Start background refresh task
        scoring_refresh_task = asyncio.create_task(refresh_cache_loop())
        logger.info("Scoring cache initialized with background refresh")
        
    except Exception as e:
        logger.error(f"Failed to initialize scoring cache: {e}")
        # Non-fatal: continue startup, cache will compute on first request
    
    yield
    
    # Shutdown
    logger.info("Shutting down Affine API server...")
    
    # Stop scoring cache refresh task
    if scoring_refresh_task:
        try:
            scoring_refresh_task.cancel()
            await scoring_refresh_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error stopping scoring cache refresh: {e}")
    
    # Stop TaskPoolManager background tasks
    if task_pool:
        try:
            await task_pool.stop_background_tasks()
            logger.info("TaskPoolManager background tasks stopped")
        except Exception as e:
            logger.error(f"Error stopping TaskPoolManager: {e}")
    
    try:
        await close_client()
        logger.info("Database client closed")
    except Exception as e:
        logger.error(f"Error closing database client: {e}")


# Create FastAPI application
app = FastAPI(
    title=config.APP_NAME,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Setup middleware
setup_middleware(app)

# Include routers with /api/v1 prefix
app.include_router(samples_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(miners_router, prefix="/api/v1")
app.include_router(scores_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
                "timestamp": int(__import__("time").time()),
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        }
    )


@app.get("/")
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn
    
    # IMPORTANT: Must use workers=1 because of TaskPoolManager:
    # - TaskPoolManager: in-memory lock mechanism (not shared across workers)
    # Using multiple workers would cause:
    # - Locks only valid within single process
    # - Inconsistency between workers
    
    uvicorn.run(
        "affine.api.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
        workers=1,  # MUST be 1 - see comment above
    )