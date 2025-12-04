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
    
    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API (e.g., "http://localhost:8000/api/v1")
            session: Shared ClientSession from GlobalSessionManager
        """
        self.base_url = base_url.rstrip("/")
        self._session = session
    
    async def close(self):
        """No-op: Session is managed by GlobalSessionManager."""
        pass
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make GET request to API endpoint.
        
        Args:
            endpoint: API endpoint path (e.g., "/miners/uid/123")
            params: Optional query parameters
            headers: Optional request headers
        
        Returns:
            Response data dict on success, None on error
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GET {url}")

        async with self._session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                result = {
                    "success": True,
                    "data": data
                }
                return data
            
            else:
                try:
                    error_detail = await response.json()
                    error_msg = error_detail.get("detail", str(error_detail))
                except:
                    error_msg = await response.text()

                result = {
                    "success": False,
                    "status_code": response.status,
                    "error": error_msg
                }
                return result

    
    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        output_json: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Make POST request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            json: Request JSON payload
            params: Optional query parameters
            headers: Optional request headers
            output_json: Whether to print JSON response to stdout
        
        Returns:
            Response data dict on success, raises exception on error
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"POST {url}")
        
        try:
            async with self._session.post(url, json=json, params=params, headers=headers) as response:
                if response.status in (200, 201):
                    response_data = await response.json()
                    result = {
                        "success": True,
                        "data": response_data
                    }
                    if output_json:
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                    return response_data
                
                else:
                    # Try to parse JSON error response
                    try:
                        error_detail = await response.json()
                        error_msg = error_detail.get("detail", str(error_detail))
                    except:
                        # Fall back to text response
                        error_msg = await response.text()
                    
                    result = {
                        "success": False,
                        "status_code": response.status,
                        "error": error_msg
                    }
                    if output_json:
                        print(json.dumps(result, indent=2, ensure_ascii=False))

                    # Raise exception for error status
                    raise Exception(f"HTTP {response.status}: {error_msg}")
        
        except Exception as e:
            result = {
                "success": False,
                "error": str(e)
            }
            if output_json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            raise


    async def put(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make PUT request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            json: Request JSON payload
            params: Optional query parameters
            headers: Optional request headers
        
        Returns:
            Response data dict on success, raises exception on error
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"PUT {url}")

        async with self._session.put(url, json=json, params=params, headers=headers) as response:
            if response.status in (200, 201, 204):
                if response.status == 204:
                    return {}
                response_data = await response.json()
                return response_data
            else:
                try:
                    error_detail = await response.json()
                    error_msg = error_detail.get("detail", str(error_detail))
                except:
                    error_msg = await response.text()
                
                raise Exception(f"HTTP {response.status}: {error_msg}")
        

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


async def get_chute_info(chute_id: str) -> Optional[Dict]:
    """Legacy function for backward compatibility.
    
    Creates a temporary APIClient to fetch chute info.
    """
    client = await create_api_client()
    try:
        return await client.get_chute_info(chute_id)
    finally:
        await client.close()


async def create_api_client(base_url: Optional[str] = None) -> APIClient:
    """Create API client with GlobalSessionManager's shared connection pool.
    
    Args:
        base_url: Custom base URL (optional, defaults to env or localhost)
    
    Returns:
        Configured APIClient instance using shared session
    """
    import os
    
    if base_url is None:
        base_url = os.getenv("API_URL", "https://api.affine.io/api/v1")
    
    # Always use GlobalSessionManager
    session = await GlobalSessionManager.get_session()
    return APIClient(base_url, session)