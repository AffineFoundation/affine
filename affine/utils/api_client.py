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
    ) -> Optional[Dict[str, Any]]:
        """Make GET request to API endpoint.
        
        Args:
            endpoint: API endpoint path (e.g., "/miners/uid/123")
            output_json: Whether to print JSON response to stdout
            exit_on_error: Whether to exit on error responses
        
        Returns:
            Response data dict on success, None on error (if exit_on_error=False)
        """
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"GET {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
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
        data: Optional[Dict[str, Any]] = None,
        output_json: bool = True,
        exit_on_error: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Make POST request to API endpoint.
        
        Args:
            endpoint: API endpoint path
            data: Request payload data
            output_json: Whether to print JSON response to stdout
            exit_on_error: Whether to exit on error responses
        
        Returns:
            Response data dict on success, None on error (if exit_on_error=False)
        """
        import aiohttp
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"POST {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status in (200, 201):
                        response_data = await response.json()
                        result = {
                            "success": True,
                            "data": response_data
                        }
                        if output_json:
                            print(json.dumps(result, indent=2, ensure_ascii=False))
                        logger.info(f"Request successful: {endpoint}")
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
                        
                        log_msg = f"HTTP {response.status}: {error_msg}"
                        if response.status == 404:
                            logger.warning(log_msg)
                        elif response.status == 429:
                            logger.warning(f"Rate limited: {error_msg}")
                        else:
                            logger.error(log_msg)
                        
                        if exit_on_error:
                            # Use os._exit() to avoid asyncio exception handling
                            sys.stdout.flush()
                            sys.stderr.flush()
                            os._exit(1)
                        return None
        
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
        base_url = os.getenv("API_URL", "http://localhost:8000/api/v1")
    
    return APIClient(base_url)