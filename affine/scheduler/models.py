import time
from dataclasses import dataclass, field
from typing import Optional
from affine.models import Miner


@dataclass
class Task:
    """Sampling task"""
    uid: int
    miner: Miner
    env_name: str
    task_id: Optional[int] = None
    seed: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SchedulerMetrics:
    """Scheduler monitoring metrics"""
    total_tasks_created: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_results_uploaded: int = 0
    active_miners: int = 0
    paused_miners: int = 0