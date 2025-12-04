"""
Executor Metrics - Data structures for tracking worker performance
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class WorkerMetrics:
    """Metrics for a worker."""
    
    worker_id: int
    env: str
    running: bool = True
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    submit_failed: int = 0
    total_execution_time: float = 0.0
    last_task_at: Optional[float] = None
    fetch_count: int = 0
    total_fetch_time: float = 0.0
    running_tasks: int = 0
    pending_tasks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)