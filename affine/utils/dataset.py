import asyncio
import base64
import json
import os
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import affine as af
from aiobotocore.session import get_session
from botocore.config import Config

class S3BufferedDataset:
    """
    Buffered dataset reader for the public 'affine-datasets' Hippius S3 bucket.
    
    This implementation uses an AUTHENTICATED aiobotocore client.
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

        short_name = self.dataset_name.split('/')[-1]

        # Configuration for the public dataset bucket
        self._bucket         = "affine-datasets"
        self._endpoint_url   = os.getenv("HIPPIUS_ENDPOINT_URL", "https://s3.hippius.com")
        self._region         = os.getenv("HIPPIUS_REGION", "decentralized")
        self._dataset_folder = f"affine/datasets/{short_name}/"
        self._index_key      = self._dataset_folder + "index.json"

        # --- AUTHENTICATION ---
        _dataset_seed_phrase = os.getenv("HIPPIUS_SEED_PHRASE")
        if not _dataset_seed_phrase or len(_dataset_seed_phrase.split()) != 12:
            raise ValueError("HIPPIUS_SEED_PHRASE environment variable is not set or is invalid.")
        
        self._access_key, self._secret_key = self._hippius_access_from_seed(_dataset_seed_phrase)

        # Internal state
        self._buffer: Deque[Any] = deque()
        self._lock   = asyncio.Lock()
        self._fill_task = None
        self._index: Optional[Dict[str, Any]] = None
        self._files: list[Dict[str, Any]] = []
        self._next_file_index: int = 0

    def _hippius_access_from_seed(self, seed: str) -> Tuple[str, str]:
        access_key = base64.b64encode(seed.encode("utf-8")).decode("utf-8")
        secret_key = seed
        return access_key, secret_key

    def _client_ctx(self):
        """Creates a properly authenticated aiobotocore client."""
        session = get_session()
        return session.create_client(
            "s3",
            endpoint_url=self._endpoint_url,
            region_name=self._region,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=Config(s3={"addressing_style": "path"}, max_pool_connections=256),
        )

    async def _ensure_index(self) -> None:
        """Fetches and parses the index.json file from the Hippius S3 bucket."""
        if self._index is not None:
            return
        af.logger.trace(f"Loading Hippius S3 index: s3://{self._bucket}/{self._index_key}")
        async with self._client_ctx() as c:
            try:
                resp = await c.get_object(Bucket=self._bucket, Key=self._index_key)
                body = await resp["Body"].read()
                self._index = json.loads(body.decode('utf-8'))
            except Exception as e:
                af.logger.error(f"Failed to fetch Hippius S3 index s3://{self._bucket}/{self._index_key}. Error: {e!r}")
                raise e
        
        self._files = list(self._index.get("files", []))
        if not self._files:
            raise RuntimeError(f"Hippius S3 index at s3://{self._bucket}/{self._index_key} contains no files.")
        self._next_file_index = 0

    async def _read_next_file(self) -> list[Any]:
        await self._ensure_index()
        if not self._files: return []
        if self._next_file_index >= len(self._files): self._next_file_index = 0
        file_info = self._files[self._next_file_index]
        self._next_file_index += 1
        key = file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
        if not key: return []
        af.logger.trace(f"Downloading Hippius S3 chunk: s3://{self._bucket}/{key}")
        async with self._client_ctx() as c:
            resp = await c.get_object(Bucket=self._bucket, Key=key)
            body = await resp["Body"].read()
        try:
            data = json.loads(body.decode('utf-8'))
        except Exception as e:
            af.logger.warning(f"Failed to parse JSON chunk {key}: {e!r}")
            return []
        return data if isinstance(data, list) else []

    async def _fill_buffer(self) -> None:
        af.logger.trace("Starting Hippius S3 buffer fill")
        while len(self._buffer) < self.buffer_size:
            try:
                rows = await self._read_next_file()
                if not rows:
                    af.logger.warning("No rows returned from Hippius S3 chunk, will retry.")
                    await asyncio.sleep(5)
                    continue
                if self.max_batch and len(rows) > self.max_batch:
                    start = random.randint(0, max(0, len(rows) - self.max_batch))
                    rows = rows[start:start + self.max_batch]
                self._buffer.extend(rows)
            except Exception as e:
                af.logger.error(f"Error in fill_buffer task: {e!r}. Retrying in 10s.")
                await asyncio.sleep(10)
        af.logger.trace("Hippius S3 buffer fill complete")

    async def get(self) -> Any:
        async with self._lock:
            if len(self._buffer) < self.buffer_size and (not self._fill_task or self._fill_task.done()):
                self._fill_task = asyncio.create_task(self._fill_buffer())
        retries = 60 
        while not self._buffer and retries > 0:
            if self._fill_task and self._fill_task.done() and self._fill_task.exception():
                raise self._fill_task.exception()
            await asyncio.sleep(0.5)
            retries -= 1
        if not self._buffer:
            raise RuntimeError("Hippius S3BufferedDataset: Buffer remained empty after waiting.")
        async with self._lock:
            item = self._buffer.popleft()
            return item

R2BufferedDataset = S3BufferedDataset
