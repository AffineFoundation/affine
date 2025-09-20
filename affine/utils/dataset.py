import asyncio
import json
import logging
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import aiohttp
import affine as af

class S3BufferedDataset:
    """
    Buffered dataset reader for the public 'affine-datasets' Hippius S3 bucket.
    
    This implementation uses aiohttp for direct, anonymous HTTP GET requests to
    completely avoid botocore's complex credential resolution and signing logic,
    providing a robust way to fetch data from a public bucket.
    """
    def __init__(
        self,
        dataset_name: str,
        buffer_size: int = 100,
        max_batch: int = 10,
        seed: Optional[int] = None,
    ):
        self.dataset_name   = dataset_name
        self.buffer_size    = buffer_size
        self.max_batch      = max_batch
        self._rng           = random.Random(seed)

        # Correctly parse the dataset name to match the bucket structure
        # e.g., "satpalsr/rl-python" -> "rl-python"
        short_name = self.dataset_name.split('/')[-1]

        # Configuration for the public training dataset bucket
        self._bucket         = "affine-datasets"
        self._endpoint_url   = "https://s3.hippius.com"
        
        # Construct the base URL for the dataset
        self._base_data_url = f"{self._endpoint_url.rstrip('/')}/{self._bucket}/affine/datasets/{short_name}"
        self._index_url      = f"{self._base_data_url}/index.json"

        # Internal state
        self._buffer: Deque[Any] = deque()
        self._lock   = asyncio.Lock()
        self._fill_task = None
        self._index: Optional[Dict[str, Any]] = None
        self._files: list[Dict[str, Any]] = []
        self._next_file_index: int = 0

    async def _http_get(self, url: str) -> bytes:
        """Performs a simple, anonymous GET request."""
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()

    async def _ensure_index(self) -> None:
        """Fetches and parses the index.json file via HTTP."""
        if self._index is not None:
            return
        
        af.logger.trace(f"Loading S3 index via HTTP from: {self._index_url}")
        try:
            body = await self._http_get(self._index_url)
            self._index = json.loads(body.decode('utf-8'))
        except aiohttp.ClientResponseError as e:
            af.logger.error(f"Failed to fetch S3 index at {self._index_url}. Status: {e.status}. Message: {e.message}. Ensure the bucket is public and the file exists.")
            raise e
        except Exception as e:
            af.logger.error(f"An unexpected error occurred while fetching S3 index at {self._index_url}.")
            raise e
        
        self._files = list(self._index.get("files", []))
        if not self._files:
            raise RuntimeError(f"S3 index at {self._index_url} contains no files.")
        self._next_file_index = 0

    async def _read_next_file(self) -> list[Any]:
        """Reads the next data chunk file specified in the index."""
        await self._ensure_index()
        if not self._files:
            return []
        
        if self._next_file_index >= len(self._files):
            self._next_file_index = 0 # Loop back to the start
        
        file_info = self._files[self._next_file_index]
        self._next_file_index += 1
        
        # Construct the full URL for the data chunk
        file_name = file_info.get("filename")
        if not file_name:
            return []
        
        chunk_url = f"{self._base_data_url}/{file_name}"
        af.logger.trace(f"Downloading S3 chunk via HTTP from: {chunk_url}")
        body = await self._http_get(chunk_url)

        try:
            data = json.loads(body.decode('utf-8'))
        except Exception as e:
            af.logger.warning(f"Failed to parse JSON chunk from {chunk_url}: {e!r}")
            return []
            
        return data if isinstance(data, list) else []

    async def _fill_buffer(self) -> None:
        """Continuously fills the internal buffer with data."""
        af.logger.trace("Starting S3 buffer fill")
        while len(self._buffer) < self.buffer_size:
            try:
                rows = await self._read_next_file()
                if not rows:
                    af.logger.warning("No rows returned from S3 chunk, will retry after a short delay.")
                    await asyncio.sleep(5)
                    continue
                
                if self.max_batch and len(rows) > self.max_batch:
                    start = self._rng.randint(0, max(0, len(rows) - self.max_batch))
                    rows = rows[start:start + self.max_batch]
                
                self._buffer.extend(rows)
            except Exception as e:
                af.logger.error(f"Error in fill_buffer task: {e!r}. Retrying in 10 seconds.")
                await asyncio.sleep(10)

        af.logger.trace("S3 buffer fill complete")

    async def get(self) -> Any:
        """Gets one item from the buffer, refilling it if necessary."""
        async with self._lock:
            if not self._buffer:
                if not self._fill_task or self._fill_task.done():
                    self._fill_task = asyncio.create_task(self._fill_buffer())
                await self._fill_task

            if not self._buffer:
                raise RuntimeError("S3BufferedDataset: failed to retrieve data from S3 after retries.")

            item = self._buffer.popleft()
            
            if len(self._buffer) < (self.buffer_size // 2) and (not self._fill_task or self._fill_task.done()):
                self._fill_task = asyncio.create_task(self._fill_buffer())
                
            return item

# Backward-compatibility alias.
R2BufferedDataset = S3BufferedDataset