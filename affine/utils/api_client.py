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

class APIClient:
    """HTTP client for Affine API requests."""
    
    def __init__(self, base_url: str):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API (e.g., "http://localhost:8000/api/v1")
        """
        self.base_url = base_url.rstrip("/")
    
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
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
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
        
        except Exception as e:
            logger.error(f"Request exception: {e}")
            result = {
                "success": False,
                "error": str(e)
            }
            return result
    
    async def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        output_json: bool = False,
        exit_on_error: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Make POST request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            json: Request JSON payload
            params: Optional query parameters
            headers: Optional request headers
            output_json: Whether to print JSON response to stdout
            exit_on_error: Whether to exit on error responses
        
        Returns:
            Response data dict on success, raises exception on error
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"POST {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=json, params=params, headers=headers) as response:
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
                        
                        if exit_on_error:
                            # Use os._exit() to avoid asyncio exception handling
                            sys.stdout.flush()
                            sys.stderr.flush()
                            os._exit(1)
                        
                        # Raise exception for error status
                        raise Exception(f"HTTP {response.status}: {error_msg}")
        
        except Exception as e:
            logger.error(f"Request exception: {e}")
            result = {
                "success": False,
                "error": str(e)
            }
            if output_json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            if exit_on_error:
                # Use os._exit() to avoid asyncio exception handling
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            
            # Re-raise exception
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
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(url, json=json, params=params, headers=headers) as response:
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
        
        except Exception as e:
            logger.error(f"PUT request exception: {e}")
            raise


async def get_chute_info(chute_id: str) -> Optional[Dict]:
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
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
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


def create_api_client(base_url: Optional[str] = None) -> APIClient:
    """Create API client with default or custom base URL.
    
    Args:
        base_url: Custom base URL (optional, defaults to env or localhost)
    
    Returns:
        Configured APIClient instance
    """
    import os
    
    if base_url is None:
        base_url = os.getenv("API_URL", "https://api.affine.io/api/v1")
    
    return APIClient(base_url)