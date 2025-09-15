import random
import asyncio
import json
import affine as af
from collections import deque
from typing import Any, Deque, List, Optional, Dict

class R2BufferedDataset:
    def __init__(
        self,
        dataset_name: str,
        total_size: int = 0,
        buffer_size: int = 100,
        max_batch: int = 10,
        seed: Optional[int] = None,
        config: str = "default",
        split: str = "train",
    ):
        self.dataset_name   = dataset_name
        self.config         = config
        self.split          = split
        self.buffer_size    = buffer_size
        self.max_batch      = max_batch
        self._rng           = random.Random(seed)

        self._buffer: Deque[Any] = deque()
        self._lock   = asyncio.Lock()
        self._fill_task = None

        # Postgres scan state
        self._db_offset: int = 0
        self.total_size = total_size

    async def _read_next_rows(self, desired: int) -> list[Any]:
        # Lazy-resolve total size from Hippius meta if not provided
        if not self.total_size:
            try:
                self.total_size = await af.get_dataset_size(
                    dataset_name=self.dataset_name,
                    config=self.config,
                    split=self.split,
                )
            except Exception:
                # Size is optional; proceed without
                pass
        # Try reading from current offset; if empty and offset > 0, wrap to 0 once
        rows: List[Any] = await af.select_dataset_rows(
            dataset_name=self.dataset_name,
            config=self.config,
            split=self.split,
            limit=desired,
            offset=self._db_offset,
            include_index=False,
        )
        if not rows and self._db_offset:
            self._db_offset = 0
            rows = await af.select_dataset_rows(
                dataset_name=self.dataset_name,
                config=self.config,
                split=self.split,
                limit=desired,
                offset=self._db_offset,
                include_index=False,
            )
        self._db_offset += len(rows)
        return rows

    async def _fill_buffer(self) -> None:
        af.logger.trace("Starting DB buffer fill")
        while len(self._buffer) < self.buffer_size:
            desired = self.max_batch if self.max_batch else (self.buffer_size - len(self._buffer))
            rows = await self._read_next_rows(desired)
            if not rows:
                break
            for item in rows:
                self._buffer.append(item)
        af.logger.trace("DB buffer fill complete")

    async def get(self) -> Any:
        # Block until there is an item available; avoids IndexError when buffers are temporarily empty
        while True:
            async with self._lock:
                if not self._fill_task or self._fill_task.done():
                    self._fill_task = asyncio.create_task(self._fill_buffer())
                if self._buffer:
                    item = self._buffer.popleft()
                    if self._fill_task.done():
                        self._fill_task = asyncio.create_task(self._fill_buffer())
                    return item
            # Wait for fill to complete or make progress
            try:
                await self._fill_task
            except Exception:
                # ignore transient fill errors and retry
                pass
            await asyncio.sleep(0.1)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.get()
