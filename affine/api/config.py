"""
API Configuration

Environment variables and settings for the API layer.
"""

import os
from typing import List


class APIConfig:
    """API configuration from environment variables."""

    # Server settings
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"


    # Rate limiting
    RATE_LIMIT_ENABLED: bool = (
        os.getenv("API_RATE_LIMIT_ENABLED", "true").lower() == "true"
    )
    RATE_LIMIT_READ: int = int(os.getenv("API_RATE_LIMIT_READ", "1000"))  # per hour
    RATE_LIMIT_WRITE: int = int(os.getenv("API_RATE_LIMIT_WRITE", "1000"))  # per hour

    # CORS
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv("API_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("API_LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("API_LOG_FILE", "")

    # Request settings
    REQUEST_TIMEOUT: int = int(os.getenv("API_REQUEST_TIMEOUT", "60"))
    MAX_REQUEST_SIZE: int = int(
        os.getenv("API_MAX_REQUEST_SIZE", str(10 * 1024 * 1024))
    )  # 10MB

    # Pagination
    DEFAULT_PAGE_SIZE: int = int(os.getenv("API_DEFAULT_PAGE_SIZE", "100"))
    MAX_PAGE_SIZE: int = int(os.getenv("API_MAX_PAGE_SIZE", "1000"))

    # Services settings
    SERVICES_ENABLED: bool = (
        os.getenv("API_SERVICES_ENABLED", "false").lower() == "true"
    )
    
    # Scheduler settings (only used if SERVICES_ENABLED=true)
    SCHEDULER_ENABLED: bool = (
        os.getenv("API_SCHEDULER_ENABLED", "false").lower() == "true"
    )
    SCHEDULER_TASK_GENERATION_INTERVAL: int = int(
        os.getenv("API_SCHEDULER_TASK_GENERATION_INTERVAL", "300")  # 5 minutes
    )
    SCHEDULER_CLEANUP_INTERVAL: int = int(
        os.getenv("API_SCHEDULER_CLEANUP_INTERVAL", "3600")  # 1 hour
    )
    SCHEDULER_MAX_TASKS_PER_MINER_ENV: int = int(
        os.getenv("API_SCHEDULER_MAX_TASKS_PER_MINER_ENV", "100")
    )

    # App metadata
    APP_NAME: str = "Affine API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "RESTful API for Affine validator infrastructure"

# Singleton instance
config = APIConfig()