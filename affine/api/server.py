"""
Affine API Server

FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from affine.api.config import config
from affine.api.middleware import setup_middleware
from affine.api.routers import (
    health_router,
    samples_router,
    tasks_router,
    miners_router,
    scores_router,
    config_router,
    logs_router,
    admin_router,
)
from affine.database import init_client, close_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Global level: INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,  # Override any existing logging configuration
)

# Set our own modules to DEBUG for detailed logging
logging.getLogger("affine").setLevel(getattr(logging, config.LOG_LEVEL.upper()))

# Set uvicorn to INFO (avoid too much noise)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

# Suppress noisy third-party loggers
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("aiobotocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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
    
    yield
    
    # Shutdown
    logger.info("Shutting down Affine API server...")
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
app.include_router(health_router, prefix="/api/v1")
app.include_router(samples_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(miners_router, prefix="/api/v1")
app.include_router(scores_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")


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
    
    uvicorn.run(
        "affine.api.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
    )