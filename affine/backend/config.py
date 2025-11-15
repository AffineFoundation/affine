"""
Dynamic Configuration System

Provides runtime configuration management with API backend and local defaults.
Follows Occam's Razor principle - simple, efficient, no unnecessary complexity.
"""

import time
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from affine.http_client import AsyncHTTPClient


@dataclass
class ConfigDefaults:
    """Code-defined default values for all configuration parameters.
    
    These defaults are used when:
    1. Database has no value set
    2. API is unreachable
    3. Fresh installation
    """
    
    # === Scheduler Configuration ===
    SCHEDULER_CHECK_INTERVAL: int = 10
    SCHEDULER_MINER_REFRESH_INTERVAL: int = 1800
    SCHEDULER_ACCELERATION_MULTIPLIER: float = 3.0
    SCHEDULER_QUEUE_PAUSE_THRESHOLD: int = 500
    SCHEDULER_QUEUE_RESUME_THRESHOLD: int = 200
    
    # === Executor Configuration ===
    EXECUTOR_WORKERS_PER_ENV: int = 10
    EXECUTOR_MAX_CONSECUTIVE_ERRORS: int = 3
    EXECUTOR_MAX_TASK_RETRIES: int = 3
    EXECUTOR_PAUSE_DURATION: int = 600
    
    # === Scorer Configuration ===
    SCORER_CALCULATION_INTERVAL: int = 1800  # 30 minutes
    SCORER_MIN_SAMPLES_PER_ENV: int = 400
    SCORER_CONFIDENCE_LEVEL: float = 0.80
    SCORER_AVERAGE_SCORE_IMPROVEMENT: float = 0.05
    
    # === Environment Configuration ===
    ENV_AFFINE_SAT_DAILY_RATE: int = 200
    ENV_AFFINE_SAT_DATASET_LENGTH: int = 200
    ENV_AFFINE_ABD_DAILY_RATE: int = 200
    ENV_AFFINE_ABD_DATASET_LENGTH: int = 200
    ENV_AFFINE_DED_DAILY_RATE: int = 200
    ENV_AFFINE_DED_DATASET_LENGTH: int = 200
    
    # === Blacklist Configuration ===
    BLACKLIST_HOTKEYS: list = field(default_factory=list)


class DynamicConfig:
    """Dynamic Configuration Manager.
    
    Manages runtime configuration with the following priority:
    1. Database value (via API) - highest priority
    2. Code default value - fallback
    
    Features:
    - 60-second cache to reduce API calls
    - Graceful degradation when API is unavailable
    - Batch operations for efficiency
    """
    
    CACHE_TTL = 60  # Cache for 60 seconds
    
    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize dynamic config manager.
        
        Args:
            api_base_url: API server URL, defaults to env var API_URL or API_BASE_URL
        """
        # Support both API_URL (Docker Compose) and API_BASE_URL (legacy)
        self.api_base_url = api_base_url or os.getenv("API_URL") or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.http_client = AsyncHTTPClient(timeout=10)
        self.defaults = ConfigDefaults()
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, timestamp)
    
    async def get(self, key: str, force_refresh: bool = False) -> Any:
        """Get configuration value.
        
        Args:
            key: Config key, e.g., "scheduler.check_interval"
            force_refresh: Force refresh from API, bypass cache
            
        Returns:
            Config value from database, or code default if not found
        """
        # Step 1: Check cache
        if not force_refresh and key in self._cache:
            value, cached_at = self._cache[key]
            if time.time() - cached_at < self.CACHE_TTL:
                return value
        
        # Step 2: Fetch from API
        try:
            url = f"{self.api_base_url}/api/v1/config/{key}"
            response = await self.http_client.get(url)
            
            if response and "param_value" in response:
                value = response["param_value"]
                
                # Update cache
                self._cache[key] = (value, time.time())
                return value
        
        except Exception:
            # API call failed, fall back to default
            pass
        
        # Step 3: Use default value
        default_value = self._get_default(key)
        
        # Cache default to avoid repeated failed API calls
        self._cache[key] = (default_value, time.time())
        return default_value
    
    async def get_batch(self, keys: list[str]) -> Dict[str, Any]:
        """Batch get configuration values.
        
        Args:
            keys: List of config keys
            
        Returns:
            Dictionary mapping keys to values
        """
        result = {}
        for key in keys:
            result[key] = await self.get(key)
        return result
    
    async def get_all(self, prefix: str = "") -> Dict[str, Any]:
        """Get all configurations, optionally filtered by prefix.
        
        Args:
            prefix: Config prefix filter, e.g., "scheduler." for all scheduler configs
            
        Returns:
            Dictionary mapping keys to values
        """
        try:
            url = f"{self.api_base_url}/api/v1/config"
            params = {"prefix": prefix} if prefix else {}
            response = await self.http_client.get(url, params=params)
            
            if response and "configs" in response:
                return response["configs"]
        
        except Exception:
            # API failed, return defaults
            pass
        
        return self._get_defaults_by_prefix(prefix)
    
    def _get_default(self, key: str) -> Any:
        """Get code-defined default value for a key.
        
        Args:
            key: Config key in dot notation (e.g., "scheduler.check_interval")
            
        Returns:
            Default value from ConfigDefaults, or None if not found
        """
        # Convert key to attribute name: "scheduler.check_interval" -> "SCHEDULER_CHECK_INTERVAL"
        attr_name = key.upper().replace(".", "_")
        
        if hasattr(self.defaults, attr_name):
            return getattr(self.defaults, attr_name)
        
        return None
    
    def _get_defaults_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get all default values matching a prefix.
        
        Args:
            prefix: Config prefix (e.g., "scheduler")
            
        Returns:
            Dictionary of matching defaults
        """
        prefix_upper = prefix.upper().replace(".", "_")
        
        result = {}
        for attr_name in dir(self.defaults):
            if not attr_name.startswith("_") and attr_name.startswith(prefix_upper):
                # Convert back to dot notation
                key = attr_name.lower().replace("_", ".")
                result[key] = getattr(self.defaults, attr_name)
        
        return result
    
    def invalidate_cache(self, key: Optional[str] = None):
        """Clear configuration cache.
        
        Args:
            key: Specific key to clear, or None to clear all cache
        """
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()
    
    async def set(self, key: str, value: Any, param_type: str, description: str = "", updated_by: str = "system") -> bool:
        """Set configuration value via API.
        
        Args:
            key: Config key
            value: Config value
            param_type: Type of value (str/int/float/bool/dict/list)
            description: Human-readable description
            updated_by: Who updated this config
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.api_base_url}/api/v1/config/{key}"
            payload = {
                "param_value": value,
                "param_type": param_type,
                "description": description,
                "updated_by": updated_by
            }
            
            response = await self.http_client.put(url, json=payload)
            
            # Invalidate cache for this key
            self.invalidate_cache(key)
            
            return response is not None
        
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete configuration (revert to default).
        
        Args:
            key: Config key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.api_base_url}/api/v1/config/{key}"
            response = await self.http_client.delete(url)
            
            # Invalidate cache
            self.invalidate_cache(key)
            
            return response is not None
        
        except Exception:
            return False


# Global singleton instance
_global_config: Optional[DynamicConfig] = None


def get_config(api_base_url: Optional[str] = None) -> DynamicConfig:
    """Get global DynamicConfig singleton instance.
    
    Args:
        api_base_url: API server URL (only used on first call)
        
    Returns:
        Global DynamicConfig instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = DynamicConfig(api_base_url)
    
    return _global_config