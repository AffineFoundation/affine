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

    # Authentication
    ADMIN_HOTKEYS: List[str] = [
        hk.strip()
        for hk in os.getenv("API_ADMIN_HOTKEYS", "").split(",")
        if hk.strip()
    ]

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = (
        os.getenv("API_RATE_LIMIT_ENABLED", "true").lower() == "true"
    )
    RATE_LIMIT_READ: int = int(os.getenv("API_RATE_LIMIT_READ", "1000"))  # per hour
    RATE_LIMIT_WRITE: int = int(os.getenv("API_RATE_LIMIT_WRITE", "100"))  # per hour
    RATE_LIMIT_ADMIN: int = int(os.getenv("API_RATE_LIMIT_ADMIN", "50"))  # per hour

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

    # App metadata
    APP_NAME: str = "Affine API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "RESTful API for Affine validator infrastructure"

    @classmethod
    def is_admin(cls, hotkey: str) -> bool:
        """Check if a hotkey has admin privileges."""
        return hotkey in cls.ADMIN_HOTKEYS


# Singleton instance
config = APIConfig()