# New content for affine/utils/dataset.py

import random
import asyncio
import json
import aiohttp
from collections import deque
from typing import Any, Deque, List, Optional, Dict
import affine as af

class R2BufferedDataset:
    def __init__(
        self,
        dataset_name: str,
        buffer_size: int = 100,
        max_batch: int = 10,
        seed: Optional[int] = None,
        # A single, CDN-fronted base URL for all shared datasets
        base_url: str = "https://datasets.affine.network", 
    ):
        if not base_url or not base_url.startswith("https://"):
            raise ValueError("A valid HTTPS base_url is required for R2BufferedDataset")

        self.dataset_name = dataset_name
        self.base_url = base_url.rstrip('/')
        self.buffer_size = buffer_size
        self.max_batch = max_batch
        self._rng = random.Random(seed)

        short_name = dataset_name.split('/')[-1]
        self._dataset_folder = f"affine/datasets/{short_name}/"
        self._index_key = self._dataset_folder + "index.json"

        self._buffer: Deque[Any] = deque()
        self._lock = asyncio.Lock()
        self._fill_task = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._index: Optional[Dict[str, Any]] = None
        self._files: list[Dict[str, Any]] = []
        self._next_file_index: int = 0
        self.total_size = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Use a TCPConnector with a high limit for concurrent fetches
            connector = aiohttp.TCPConnector(limit_per_host=64)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _fetch_json(self, path: str) -> Optional[Any]:
        """Helper to fetch and parse JSON from a relative path via HTTP."""
        url = f"{self.base_url}/{path}"
        af.logger.trace(f"Fetching from public CDN: {url}")
        session = await self._get_session()
        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    af.logger.warning(f"Failed to fetch {url}, status: {resp.status}, body: {await resp.text()}")
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            af.logger.warning(f"Error fetching/parsing {url}: {e}")
            return None

    async def _ensure_index(self) -> bool:
        if self._index is not None:
            return True
        
        index_data = await self._fetch_json(self._index_key)
        if index_data is None:
            af.logger.error(f"Could not load R2 index from {self.base_url}/{self._index_key}")
            return False

        self._index = index_data
        self._files = list(self._index.get("files", []))
        self.total_size = int(self._index.get("total_rows", 0))
        self._rng.shuffle(self._files)
        
        if not self._files:
            af.logger.error("R2 index contains no files")
            return False
            
        self._next_file_index = 0
        return True

    async def _fill_buffer(self) -> None:
        af.logger.trace("Starting R2 buffer fill via public URL")
        if not await self._ensure_index():
            return

        # Continue filling until buffer is full
        while len(self._buffer) < self.buffer_size:
            if self._next_file_index >= len(self._files):
                af.logger.info("Completed one pass of all dataset files. Looping.")
                self._next_file_index = 0
                if not self._files: break

            file_info = self._files[self._next_file_index]
            self._next_file_index += 1
            
            key = file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
            if not key: continue

            rows = await self._fetch_json(key)
            if isinstance(rows, list):
                self._rng.shuffle(rows)
                for item in rows:
                    self._buffer.append(item)
        af.logger.trace("R2 buffer fill complete")

    async def get(self) -> Any:
        async with self._lock:
            if not self._buffer and (not self._fill_task or self._fill_task.done()):
                self._fill_task = asyncio.create_task(self._fill_buffer())

            if not self._buffer:
                await self._fill_task

            if not self._buffer:
                raise StopAsyncIteration("Dataset buffer is empty and could not be refilled.")
            
            item = self._buffer.popleft()
            
            if len(self._buffer) < self.buffer_size // 2 and (not self._fill_task or self._fill_task.done()):
                self._fill_task = asyncio.create_task(self._fill_buffer())
            return item

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.get()
