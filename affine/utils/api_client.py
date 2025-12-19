"""
API Client Utility

Provides a reusable HTTP client for making API requests to the Affine API server.
Handles common patterns like error handling, JSON response parsing, and logging.
"""

import json
import sys
import os
from typing import Optional, Dict, Any
from affine.core.setup import logger
import aiohttp
import asyncio
from affine.utils.errors import NetworkError, ApiResponseError
from affine.config import config as affine_config, ConfigError


class GlobalSessionManager:
    """Singleton manager for shared aiohttp ClientSession across all workers.
    
    This ensures all HTTP requests share a single connection pool, minimizing
    file descriptor usage and improving performance.
    """
    
    _instance: Optional['GlobalSessionManager'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    _session: Optional[aiohttp.ClientSession] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        """Get or create the global shared session.
        
        Returns:
            Shared ClientSession instance
        """
        async with cls._lock:
            if cls._session is None or cls._session.closed:
                # Configure connector for shared connection pool
                # Need 8 workers × 60 concurrent tasks = 480 potential connections
                connector = aiohttp.TCPConnector(
                    limit=1000,  # Increased to handle 8 workers × 60 tasks
                    limit_per_host=0,  # No per-host limit (use total limit)
                    ttl_dns_cache=300,  # DNS cache TTL
                    force_close=False,  # Allow connection reuse
                    enable_cleanup_closed=True,  # Clean up closed connections
                    keepalive_timeout=30,  # Close idle connections after 30s (prevent stale connections)
                )
                
                # Increase connection timeout to handle pool contention
                timeout = aiohttp.ClientTimeout(
                    total=None,  # No total timeout
                    connect=60,  # 60s connection timeout (wait for available connection)
                    sock_read=None  # No read timeout
                )
                
                cls._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    connector_owner=True  # Ensure connector is closed with session
                )
                
            
            return cls._session
    
    @classmethod
    async def close(cls):
        """Close the global shared session."""
        async with cls._lock:
            if cls._session and not cls._session.closed:
                await cls._session.close()
                cls._session = None
                logger.info("GlobalSessionManager: Closed shared session")

class APIClient:
    """HTTP client for Affine API requests.
    
    Uses GlobalSessionManager's shared connection pool for all requests.
    """
    
    def __init__(self, base_url: str, session: aiohttp.ClientSession, timeout: int = 30, retry_config: Any = None):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API (e.g., "http://localhost:8000/api/v1")
            session: Shared ClientSession from GlobalSessionManager
            timeout: Request timeout in seconds
            retry_config: Retry configuration object
        """
        self.base_url = base_url.rstrip("/")
        self._session = session
        self.timeout = timeout
        self.retry_config = retry_config
    
    async def close(self):
        """No-op: Session is managed by GlobalSessionManager."""
        pass
    
    async def _request_with_retry(self, method: str, url: str, **kwargs) -> Any:
        """Execute request with retry logic."""
        max_attempts = 1
        backoff = 1.0
        
        if self.retry_config:
            max_attempts = self.retry_config.max_attempts
            backoff = self.retry_config.backoff_seconds
            
        kwargs.setdefault('timeout', self.timeout)
        
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                async with self._session.request(method, url, **kwargs) as response:
                    # Note: We consider 5xx errors retryable. 4xx are client errors generally not retryable (except maybe 429?)
                    # For now, let's stick to 5xx being retryable if needed, or just connection errors.
                    # Actually standard practice: connection errors + 5xx.
                    
                    if response.status >= 500:
                         error_text = await response.text()
                         msg = f"Server Error {response.status}: {error_text[:200]}"
                         if attempt == max_attempts - 1:
                              raise ApiResponseError(msg, response.status, url, error_text)
                    elif response.status >= 400:
                        body = await response.text()
                        raise ApiResponseError(f"HTTP {response.status}: {body[:200]}", response.status, url, body)
                    else:
                        # Success 2xx
                         if response.status == 204:
                             return {}
                         try:
                             return await response.json()
                         except Exception:
                             raw = await response.text()
                             raise ApiResponseError(f"Invalid JSON response: {raw[:200]}", response.status, url, raw)
                    
                    # If we got here with 500+ and didn't raise
                    logger.warning(f"Request {method} {url} failed with status {response.status}. Retrying in {backoff}s...")
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt == max_attempts - 1:
                    raise NetworkError(f"Network error during {method} {url}: {e}", url, e)
                logger.warning(f"Request {method} {url} failed: {e}. Retrying in {backoff}s...")
            
            await asyncio.sleep(backoff)
            backoff *= 2.0
            
        if last_error:
            raise NetworkError(f"Max retries exceeded for {method} {url}", url, last_error)
        raise NetworkError(f"Unknown error during {method} {url}", url, None)


    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make GET request to API endpoint."""
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GET {url}")
        return await self._request_with_retry("GET", url, params=params, headers=headers)

    
    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        output_json: bool = False,
    ) -> Any:
        """Make POST request to API endpoint."""
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"POST {url}")
        
        try:
            return await self._request_with_retry("POST", url, json=json, params=params, headers=headers)
        except Exception as e:
            if output_json:
                 msg = str(e)
                 if isinstance(e, ApiResponseError):
                      try:
                          import json as json_lib
                          err_json = json_lib.loads(e.response_body)
                          msg = err_json.get("detail", msg)
                      except:
                          pass
                 print(f'{{"success": false, "error": "{msg}"}}')
            raise

    async def put(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make PUT request to API endpoint."""
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"PUT {url}")
        return await self._request_with_retry("PUT", url, json=json, params=params, headers=headers)
        

    async def get_chute_info(self, chute_id: str) -> Optional[Dict]:
        """Get chute info from Chutes API.
        
        Args:
            chute_id: Chute deployment ID
            
        Returns:
            Chute info dict or None if failed
        """
        url = f"https://api.chutes.ai/chutes/{chute_id}"
        token = os.getenv("CHUTES_API_KEY", "")
        
        if not token:
            logger.warning("CHUTES_API_KEY not configured")
            return None
        
        headers = {"Authorization": token}
        
        try:
            async with self._session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                
                info = await resp.json()
                # Remove unnecessary fields
                for k in ("readme", "cords", "tagline", "instances"):
                    info.pop(k, None)
                info.get("image", {}).pop("readme", None)
                
                return info
        except Exception as e:
            logger.debug(f"Failed to fetch chute {chute_id}: {e}")
            return None


class CLIAPIClient:
    """CLI-specific API client context manager.
    
    Creates an independent session for one-time CLI commands,
    automatically closing it when done. This is separate from
    long-running services that use GlobalSessionManager.
    
    Usage:
        async with cli_api_client() as client:
            data = await client.get("/miners/uid/42")
    """
    
    def __init__(self, base_url: Optional[str] = None):
        try:
            # Attempt to resolve profile
            profile = affine_config.get_profile()
            self.base_url = base_url or profile.base_url
            self.timeout = profile.timeout
            self.retry_config = profile.retry
        except ConfigError:
            # If no config is present but base_url is provided manually (CLI override behavior), we use it.
            # But get_profile raises if no profile is found or if default is missing.
            # We want to support manual override if config is broken/missing?
            # User req: "If resolved profile does not exist, raise a clear error."
            # So raising is correct.
            if base_url:
                self.base_url = base_url
                self.timeout = 30
                self.retry_config = None
            else:
                raise

        self._session: Optional[aiohttp.ClientSession] = None
        self._client: Optional['APIClient'] = None
    
    async def __aenter__(self) -> 'APIClient':
        """Enter context: create independent session and client"""
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=0,
            force_close=False,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=30,  # CLI doesn't need long connection timeout
            sock_read=None
        )
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True
        )
        
        self._client = APIClient(self.base_url, self._session, timeout=self.timeout, retry_config=self.retry_config)
        return self._client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context: close session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("CLIAPIClient: Closed independent session")
        return False  # Don't suppress exceptions


def cli_api_client(base_url: Optional[str] = None) -> CLIAPIClient:
    """Create CLI-specific API client context manager.
    
    Args:
        base_url: Custom base URL (optional)
    
    Returns:
        CLIAPIClient context manager
        
    Example:
        async with cli_api_client() as client:
            data = await client.get("/miners/uid/42")
            print(json.dumps(data, indent=2))
    """
    return CLIAPIClient(base_url)


async def get_chute_info(chute_id: str) -> Optional[Dict]:
    """Legacy function for backward compatibility.
    
    Creates a temporary APIClient to fetch chute info.
    """
    async with cli_api_client() as client:
        return await client.get_chute_info(chute_id)


async def create_api_client(base_url: Optional[str] = None) -> APIClient:
    """Create API client with GlobalSessionManager's shared connection pool.
    
    Args:
        base_url: Custom base URL (optional, defaults to config)
    
    Returns:
        Configured APIClient instance using shared session
    """
    
    # Load from config
    profile = affine_config.get_profile()
    
    url = base_url or profile.base_url
    
    # Always use GlobalSessionManager
    session = await GlobalSessionManager.get_session()
    
    return APIClient(url, session, timeout=profile.timeout, retry_config=profile.retry)