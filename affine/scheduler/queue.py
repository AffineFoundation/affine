import asyncio
import time
from typing import Dict, List
from affine.scheduler.models import Task
from affine.setup import logger


class TaskQueue:
    """Task queue with backpressure control and fair scheduling"""
    
    def __init__(
        self,
        max_size: int = 10000,
        warning_threshold: int = 5000,
        pause_threshold: int = 8000,
        resume_threshold: int = 3000,
        batch_size: int = 50,
    ):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.warning_threshold = warning_threshold
        self.pause_threshold = pause_threshold
        self.resume_threshold = resume_threshold
        self.batch_size = batch_size
        
        self.is_paused = False
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        
        self.batch_buffer: Dict[int, List[Task]] = {}
        self.batch_lock = asyncio.Lock()
        
        self.total_enqueued = 0
        self.total_dequeued = 0
        self._last_warning_time = 0
    
    async def put(self, task: Task, sampler_id: int):
        """Add task with batch buffering for fairness"""
        async with self.batch_lock:
            if sampler_id not in self.batch_buffer:
                self.batch_buffer[sampler_id] = []
            
            self.batch_buffer[sampler_id].append(task)
            
            if len(self.batch_buffer[sampler_id]) >= self.batch_size:
                await self._flush_batch(sampler_id)
    
    async def _flush_batch(self, sampler_id: int):
        """Flush batch to queue after waiting for backpressure to clear"""
        await self.pause_event.wait()
        
        batch = self.batch_buffer.get(sampler_id, [])
        if not batch:
            return
        
        for task in batch:
            await self.queue.put(task)
            self.total_enqueued += 1
        
        self.batch_buffer[sampler_id] = []
        self._check_backpressure()
    
    async def flush_all_batches(self):
        """Periodically flush incomplete batches to avoid starvation"""
        async with self.batch_lock:
            for sampler_id in list(self.batch_buffer.keys()):
                if self.batch_buffer[sampler_id]:
                    await self._flush_batch(sampler_id)
    
    def _check_backpressure(self):
        """Monitor queue size and trigger backpressure"""
        size = self.queue.qsize()
        now = time.monotonic()
        
        if size >= self.pause_threshold and not self.is_paused:
            self.is_paused = True
            self.pause_event.clear()
            logger.warning(f"[TaskQueue] PAUSED: size {size} >= {self.pause_threshold}")
        
        elif size <= self.resume_threshold and self.is_paused:
            self.is_paused = False
            self.pause_event.set()
            logger.info(f"[TaskQueue] RESUMED: size {size} <= {self.resume_threshold}")
        
        elif size >= self.warning_threshold and not self.is_paused:
            if now - self._last_warning_time >= 60:
                logger.warning(f"[TaskQueue] WARNING: size {size} >= {self.warning_threshold}")
                self._last_warning_time = now
    
    async def get(self) -> Task:
        """Get task from queue"""
        task = await self.queue.get()
        self.total_dequeued += 1
        self._check_backpressure()
        return task
    
    def qsize(self) -> int:
        return self.queue.qsize()