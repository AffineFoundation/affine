"""File-based queue for model verification tasks."""

import os
import time
import asyncio
import orjson
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from affine.setup import logger


class TaskStatus(Enum):
    """Task status enum."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VerificationTask:
    """Verification task for a model."""

    task_id: str
    block: int
    hotkey: str
    model: str
    revision: str
    timestamp: float
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0  # Higher priority = processed first
    error: Optional[str] = None
    retry_count: int = 0

    @classmethod
    def create(
        cls,
        block: int,
        hotkey: str,
        model: str,
        revision: str,
        priority: int = 0,
    ) -> "VerificationTask":
        """Create a new verification task."""
        task_id = f"block_{block}_hotkey_{hotkey[:8]}_{int(time.time())}"
        return cls(
            task_id=task_id,
            block=block,
            hotkey=hotkey,
            model=model,
            revision=revision,
            timestamp=time.time(),
            priority=priority,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "VerificationTask":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = TaskStatus(data["status"])
        return cls(**data)


class VerificationQueue:
    """File-based queue for verification tasks."""

    def __init__(self, queue_dir: Optional[Path] = None):
        """Initialize queue.

        Args:
            queue_dir: Directory to store queue files
        """
        if queue_dir is None:
            queue_dir = Path(
                os.getenv(
                    "AFFINE_CACHE_DIR",
                    Path.home() / ".cache" / "affine" / "verification",
                )
            ) / "queue"

        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Queue file paths
        self.pending_file = self.queue_dir / "pending.json"
        self.processing_file = self.queue_dir / "processing.json"
        self.completed_file = self.queue_dir / "completed.json"
        self.failed_file = self.queue_dir / "failed.json"

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info(f"Initialized verification queue at {self.queue_dir}")

    async def enqueue(self, task: VerificationTask) -> None:
        """Add a task to the pending queue.

        Args:
            task: Verification task to add
        """
        async with self._lock:
            # Load existing pending tasks
            pending = await self._load_queue(self.pending_file)

            # Check if task already exists
            if any(t["task_id"] == task.task_id for t in pending):
                logger.debug(f"Task {task.task_id} already in queue, skipping")
                return

            # Add new task
            pending.append(task.to_dict())

            # Sort by priority (higher first) then by timestamp (older first)
            pending.sort(key=lambda t: (-t["priority"], t["timestamp"]))

            # Save queue
            await self._save_queue(self.pending_file, pending)

            logger.info(f"Enqueued task: {task.task_id} (priority={task.priority})")

    async def dequeue(self) -> Optional[VerificationTask]:
        """Get the next task from the pending queue and move to processing.

        Returns:
            Next verification task or None if queue is empty
        """
        async with self._lock:
            # Load pending tasks
            pending = await self._load_queue(self.pending_file)

            if not pending:
                return None

            # Get the first task (highest priority, oldest)
            task_data = pending.pop(0)
            task = VerificationTask.from_dict(task_data)

            # Update status
            task.status = TaskStatus.PROCESSING

            # Save updated pending queue
            await self._save_queue(self.pending_file, pending)

            # Add to processing queue
            processing = await self._load_queue(self.processing_file)
            processing.append(task.to_dict())
            await self._save_queue(self.processing_file, processing)

            logger.info(f"Dequeued task: {task.task_id}")
            return task

    async def complete_task(self, task_id: str) -> None:
        """Mark a task as completed.

        Args:
            task_id: Task ID to mark as completed
        """
        async with self._lock:
            # Find and remove from processing
            processing = await self._load_queue(self.processing_file)
            task_data = None

            for i, t in enumerate(processing):
                if t["task_id"] == task_id:
                    task_data = processing.pop(i)
                    break

            if task_data is None:
                logger.warning(f"Task {task_id} not found in processing queue")
                return

            # Update status
            task = VerificationTask.from_dict(task_data)
            task.status = TaskStatus.COMPLETED

            # Save updated processing queue
            await self._save_queue(self.processing_file, processing)

            # Add to completed queue
            completed = await self._load_queue(self.completed_file)
            completed.append(task.to_dict())

            # Keep only last 1000 completed tasks
            if len(completed) > 1000:
                completed = completed[-1000:]

            await self._save_queue(self.completed_file, completed)

            logger.info(f"Completed task: {task_id}")

    async def fail_task(self, task_id: str, error: str, retry: bool = True) -> None:
        """Mark a task as failed.

        Args:
            task_id: Task ID to mark as failed
            error: Error message
            retry: Whether to retry the task
        """
        async with self._lock:
            # Find and remove from processing
            processing = await self._load_queue(self.processing_file)
            task_data = None

            for i, t in enumerate(processing):
                if t["task_id"] == task_id:
                    task_data = processing.pop(i)
                    break

            if task_data is None:
                logger.warning(f"Task {task_id} not found in processing queue")
                return

            # Update task
            task = VerificationTask.from_dict(task_data)
            task.error = error
            task.retry_count += 1

            # Save updated processing queue
            await self._save_queue(self.processing_file, processing)

            # Decide whether to retry or mark as failed
            max_retries = int(os.getenv("VERIFICATION_MAX_RETRIES", "3"))
            if retry and task.retry_count < max_retries:
                # Re-add to pending queue with lower priority
                task.status = TaskStatus.PENDING
                task.priority = max(0, task.priority - 1)

                pending = await self._load_queue(self.pending_file)
                pending.append(task.to_dict())
                pending.sort(key=lambda t: (-t["priority"], t["timestamp"]))
                await self._save_queue(self.pending_file, pending)

                logger.warning(
                    f"Task {task_id} failed (retry {task.retry_count}/{max_retries}): {error}"
                )
            else:
                # Mark as failed
                task.status = TaskStatus.FAILED

                failed = await self._load_queue(self.failed_file)
                failed.append(task.to_dict())

                # Keep only last 1000 failed tasks
                if len(failed) > 1000:
                    failed = failed[-1000:]

                await self._save_queue(self.failed_file, failed)

                logger.error(f"Task {task_id} failed permanently: {error}")

    async def get_queue_stats(self) -> Dict:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        async with self._lock:
            pending = await self._load_queue(self.pending_file)
            processing = await self._load_queue(self.processing_file)
            completed = await self._load_queue(self.completed_file)
            failed = await self._load_queue(self.failed_file)

            return {
                "pending": len(pending),
                "processing": len(processing),
                "completed": len(completed),
                "failed": len(failed),
            }

    async def get_pending_tasks(self) -> List[VerificationTask]:
        """Get all pending tasks.

        Returns:
            List of pending tasks
        """
        async with self._lock:
            pending = await self._load_queue(self.pending_file)
            return [VerificationTask.from_dict(t) for t in pending]

    async def clear_completed(self) -> None:
        """Clear completed tasks."""
        async with self._lock:
            await self._save_queue(self.completed_file, [])
            logger.info("Cleared completed tasks")

    async def clear_failed(self) -> None:
        """Clear failed tasks."""
        async with self._lock:
            await self._save_queue(self.failed_file, [])
            logger.info("Cleared failed tasks")

    async def _load_queue(self, file_path: Path) -> List[Dict]:
        """Load queue from file.

        Args:
            file_path: Path to queue file

        Returns:
            List of task dictionaries
        """
        if not file_path.exists():
            return []

        try:
            data = orjson.loads(file_path.read_bytes())
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to load queue from {file_path}: {e}")
            return []

    async def _save_queue(self, file_path: Path, tasks: List[Dict]) -> None:
        """Save queue to file.

        Args:
            file_path: Path to queue file
            tasks: List of task dictionaries
        """
        try:
            file_path.write_bytes(orjson.dumps(tasks, option=orjson.OPT_INDENT_2))
        except Exception as e:
            logger.error(f"Failed to save queue to {file_path}: {e}")
            raise
