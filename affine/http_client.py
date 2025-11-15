import os
import atexit
import asyncio
import aiohttp
from typing import Dict

_HTTP_SEMS: Dict[int, asyncio.Semaphore] = {}
_CLIENTS: Dict[int, aiohttp.ClientSession] = {}

async def _cleanup_clients():
    for client in _CLIENTS.values():
        if client and not client.closed:
            await client.close()
    _CLIENTS.clear()

def _sync_cleanup():
    try:
        asyncio.run(_cleanup_clients())
    except RuntimeError:
        pass

atexit.register(_sync_cleanup)

async def _get_sem() -> asyncio.Semaphore:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    sem = _HTTP_SEMS.get(key)
    if sem is None:
        sem = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "400")))
        _HTTP_SEMS[key] = sem
    return sem

async def _get_client() -> aiohttp.ClientSession:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    client = _CLIENTS.get(key)
    if client is None or client.closed:
        limit = int(os.getenv("AFFINE_HTTP_CONCURRENCY", "400"))
        conn = aiohttp.TCPConnector(
            limit=limit,
            limit_per_host=0,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        client = aiohttp.ClientSession(
            connector=conn,
            timeout=aiohttp.ClientTimeout(total=None)
        )
        _CLIENTS[key] = client
    return client


class AsyncHTTPClient:
    """Async HTTP Client with timeout and retry support."""
    
    def __init__(self, timeout: int = 30):
        """Initialize async HTTP client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def get(self, url: str, params: dict = None, headers: dict = None) -> dict:
        """GET request.
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response JSON data
        """
        client = await _get_client()
        async with client.get(url, params=params, headers=headers, timeout=self.timeout) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post(self, url: str, json: dict = None, headers: dict = None, params: dict = None) -> dict:
        """POST request.
        
        Args:
            url: Request URL
            json: JSON body
            headers: Request headers
            params: Query parameters
            
        Returns:
            Response JSON data
        """
        client = await _get_client()
        async with client.post(url, json=json, headers=headers, params=params, timeout=self.timeout) as response:
            response.raise_for_status()
            return await response.json()
    
    async def put(self, url: str, json: dict = None, headers: dict = None) -> dict:
        """PUT request.
        
        Args:
            url: Request URL
            json: JSON body
            headers: Request headers
            
        Returns:
            Response JSON data
        """
        client = await _get_client()
        async with client.put(url, json=json, headers=headers, timeout=self.timeout) as response:
            response.raise_for_status()
            return await response.json()
    
    async def delete(self, url: str, headers: dict = None) -> dict:
        """DELETE request.
        
        Args:
            url: Request URL
            headers: Request headers
            
        Returns:
            Response JSON data
        """
        client = await _get_client()
        async with client.delete(url, headers=headers, timeout=self.timeout) as response:
            response.raise_for_status()
            return await response.json()