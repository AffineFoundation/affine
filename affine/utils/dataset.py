import random
import asyncio
import logging
import os
import json
import aiohttp
from botocore.config import Config
from aiobotocore.session import get_session
import affine as af
from collections import deque
from typing import Any, Deque, List, Optional, Dict

# class BufferedDataset:
#     def __init__(
#         self,
#         dataset_name: str,
#         total_size: int,
#         buffer_size: int = 100,
#         max_batch: int = 10,
#         seed: Optional[int] = None,
#         split: str = "train",
#         config: str = "default",
#         max_retries: int = 5,
#         initial_backoff: float = 10.0,
#         backoff_factor: float = 2.0,
#         max_backoff: float = 30.0
#     ):
#         self.dataset_name   = dataset_name
#         self.total_size     = total_size
#         self.buffer_size    = buffer_size
#         self.max_batch      = max_batch
#         self.split          = split
#         self.config         = config
#         self.max_retries    = max_retries
#         self.initial_backoff= initial_backoff
#         self.backoff_factor = backoff_factor
#         self.max_backoff    = max_backoff

#         self._buffer: Deque[Any] = deque()
#         self._lock   = asyncio.Lock()
#         self._fill_task = None
#         self._rng    = random.Random(seed)

#     async def fetch_hf(self, offset: int, length: int) -> List[Any]:
#         url = (
#             f"https://datasets-server.huggingface.co/rows?"
#             f"dataset={self.dataset_name}"
#             f"&config={self.config}"
#             f"&split={self.split}"
#             f"&offset={offset}"
#             f"&length={length}"
#         )
#         backoff = self.initial_backoff
#         for attempt in range(1, self.max_retries + 1):
#             af.logger.trace(f"HF fetch attempt {attempt}: offset={offset}, len={length}")
#             async with aiohttp.ClientSession() as sess:
#                 async with sess.get(url, timeout=30) as resp:
#                     if resp.status == 429:
#                         af.logger.warning(f"Ratelimit hit; sleeping {backoff:.1f}s")
#                         await asyncio.sleep(backoff)
#                         backoff = min(backoff * self.backoff_factor, self.max_backoff)
#                         continue
#                     resp.raise_for_status()
#                     data = await resp.json()
#                     rows = [r["row"] for r in data.get("rows", [])]
#                     af.logger.trace(f"Fetched {len(rows)} rows")
#                     return rows
#         raise RuntimeError("HF rate-limit retries exhausted")

#     async def _fill_buffer(self) -> None:
#         af.logger.trace("Starting buffer fill")
#         while len(self._buffer) < self.buffer_size:
#             batch = self.max_batch
#             offset = self._rng.randint(0, max(0, self.total_size - batch))
#             try:
#                 rows = await self.fetch_hf(offset, batch)
#             except Exception as e:
#                 af.logger.warning(f"Fetch error: {e!r}")
#                 continue
#             for item in rows:
#                 self._buffer.append(item)
#         af.logger.trace("Buffer fill complete")

#     async def get(self) -> Any:
#         async with self._lock:
#             if not self._fill_task or self._fill_task.done():
#                 self._fill_task = asyncio.create_task(self._fill_buffer())
#             if not self._buffer:
#                 await self._fill_task
#             item = self._buffer.popleft()
#             if self._fill_task.done():
#                 self._fill_task = asyncio.create_task(self._fill_buffer())
#             return item

#     def __aiter__(self):
#         return self

#     async def __anext__(self):
#         return await self.get()


class R2BufferedDataset:
    def __init__(
        self,
        dataset_name: str,
        total_size: int = 0,
        buffer_size: int = 100,
        max_batch: int = 10,
        seed: Optional[int] = None,
    ):
        self.dataset_name   = dataset_name
        self.buffer_size    = buffer_size
        self.max_batch      = max_batch
        self._rng           = random.Random(seed)

        short_name          = dataset_name
        self._dataset_folder= f"affine/datasets/{short_name}/"
        self._index_key     = self._dataset_folder + "index.json"

        self._folder        = af.FOLDER
        bucket_id           = af.BUCKET
        endpoint            = af.ENDPOINT
        access_key          = af.ACCESS
        secret_key          = af.SECRET

        self._endpoint_url  = endpoint
        self._access_key    = access_key
        self._secret_key    = secret_key

        self._buffer: Deque[Any] = deque()
        self._lock   = asyncio.Lock()
        self._fill_task = None

        self._index: Optional[Dict[str, Any]] = None
        self._files: list[Dict[str, Any]] = []
        self._next_file_index: int = 0
        self.total_size = total_size

    def _client_ctx(self):
        if not self._endpoint_url:
            raise RuntimeError("R2 endpoint is not configured (missing R2_BUCKET_ID)")
        sess = get_session()
        return sess.create_client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=Config(max_pool_connections=256),
        )

    async def _ensure_index(self) -> None:
        if self._index is not None:
            return
        af.logger.trace(f"Loading R2 index: s3://{self._folder}/{self._index_key}")
        async with self._client_ctx() as c:
            resp = await c.get_object(Bucket=self._folder, Key=self._index_key)
            body = await resp["Body"].read()
            self._index = json.loads(body.decode())
        self._files = list(self._index.get("files", []))
        if not self.total_size:
            self.total_size = int(self._index.get("total_rows", 0))
        if not self._files:
            raise RuntimeError("R2 index contains no files")
        self._next_file_index = 0

    async def _read_next_file(self) -> list[Any]:
        await self._ensure_index()
        if not self._files:
            return []
        if self._next_file_index >= len(self._files):
            self._next_file_index = 0
        file_info = self._files[self._next_file_index]
        self._next_file_index += 1
        key = file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
        if not key:
            return []
        af.logger.trace(f"Downloading R2 chunk: s3://{self._folder}/{key}")
        async with self._client_ctx() as c:
            resp = await c.get_object(Bucket=self._folder, Key=key)
            body = await resp["Body"].read()
        try:
            data = json.loads(body.decode())
        except Exception as e:
            af.logger.warning(f"Failed to parse chunk {key}: {e!r}")
            return []
        if not isinstance(data, list):
            return []
        return data

    async def _fill_buffer(self) -> None:
        af.logger.trace("Starting R2 buffer fill")
        while len(self._buffer) < self.buffer_size:
            rows = await self._read_next_file()
            if not rows:
                break
            if self.max_batch and len(rows) > self.max_batch:
                start = self._rng.randint(0, max(0, len(rows) - self.max_batch))
                rows = rows[start:start + self.max_batch]
            for item in rows:
                self._buffer.append(item)
        af.logger.trace("R2 buffer fill complete")

    async def get(self) -> Any:
        async with self._lock:
            if not self._fill_task or self._fill_task.done():
                self._fill_task = asyncio.create_task(self._fill_buffer())
            if not self._buffer:
                await self._fill_task
            item = self._buffer.popleft()
            if self._fill_task.done():
                self._fill_task = asyncio.create_task(self._fill_buffer())
            return item

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.get()
