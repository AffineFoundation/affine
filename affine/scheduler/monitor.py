#!/usr/bin/env python3
"""
Scheduler monitoring and status API

Provides comprehensive monitoring of the sampling scheduler including:
- Per-miner sampling statistics
- Queue status and throughput
- Worker utilization
- Error tracking
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from affine.scheduler.models import Task
from affine.models import Result


@dataclass
class MinerSamplingStats:
    """Statistics for a single miner"""
    uid: int
    hotkey: str
    model: str
    
    # Status
    status: str  # "active", "paused", "stopped"
    pause_until: float = 0.0
    pause_reason: Optional[str] = None
    
    # Sampling rates
    rate_multiplier: float = 1.0
    configured_daily_rate: int = 200
    effective_daily_rate: float = 0.0
    
    # Counters (last 1 hour)
    samples_1h: Dict[str, int] = field(default_factory=dict)
    total_samples_1h: int = 0
    
    # Counters (last 24 hours)
    samples_24h: Dict[str, int] = field(default_factory=dict)
    total_samples_24h: int = 0
    
    # Error tracking
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    error_count_1h: int = 0


@dataclass
class QueueStats:
    """Queue statistics"""
    current_size: int
    max_size: int
    warning_threshold: int
    pause_threshold: int
    resume_threshold: int
    
    # Status
    is_paused: bool = False
    is_warning: bool = False
    
    # Throughput (tasks/minute)
    enqueue_rate: float = 0.0
    dequeue_rate: float = 0.0
    
    # Totals
    total_enqueued: int = 0
    total_dequeued: int = 0
    
    # Per-environment breakdown
    env_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class WorkerStats:
    """Worker pool statistics"""
    total_workers: int
    active_workers: int
    idle_workers: int
    
    # Throughput
    tasks_per_minute: float = 0.0
    avg_task_duration: float = 0.0
    
    # Utilization (0-1)
    utilization: float = 0.0


@dataclass
class SchedulerStatus:
    """Complete scheduler status"""
    timestamp: float
    uptime_seconds: float
    
    # Summary
    total_miners: int
    active_miners: int
    paused_miners: int
    
    # Queue
    queue: QueueStats
    
    # Workers
    workers: WorkerStats
    
    # Detailed miner stats
    miners: List[MinerSamplingStats] = field(default_factory=list)
    
    # Environment stats
    env_stats: Dict[str, Dict] = field(default_factory=dict)


class SchedulerMonitor:
    """Monitor and track scheduler statistics"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.start_time = time.time()
        
        # Time-windowed counters
        self.sample_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=3600))
        self.error_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=3600))
        
        # Throughput tracking
        self.enqueue_history = deque(maxlen=60)  # Last 60 seconds
        self.dequeue_history = deque(maxlen=60)
        self.last_enqueue_count = 0
        self.last_dequeue_count = 0
        
        # Worker tracking
        self.task_completion_times = deque(maxlen=100)
    
    def record_sample(self, uid: int, env_name: str):
        """Record a sample event"""
        now = time.time()
        self.sample_history[uid].append({
            'time': now,
            'env': env_name,
        })
    
    def record_error(self, uid: int, error: str):
        """Record an error event"""
        now = time.time()
        self.error_history[uid].append({
            'time': now,
            'error': error,
        })
    
    def record_task_completion(self, duration: float):
        """Record task completion time"""
        self.task_completion_times.append(duration)
    
    def _get_miner_stats(self, uid: int) -> MinerSamplingStats:
        """Get statistics for a single miner"""
        sampler = self.scheduler.samplers.get(uid)
        if not sampler:
            return None
        
        now = time.time()
        
        # Calculate time windows
        one_hour_ago = now - 3600
        one_day_ago = now - 86400
        
        # Count samples in time windows
        samples_1h = defaultdict(int)
        samples_24h = defaultdict(int)
        
        for event in self.sample_history[uid]:
            if event['time'] >= one_hour_ago:
                samples_1h[event['env']] += 1
            if event['time'] >= one_day_ago:
                samples_24h[event['env']] += 1
        
        # Count errors in last hour
        error_count_1h = sum(
            1 for e in self.error_history[uid]
            if e['time'] >= one_hour_ago
        )
        
        # Get last error
        last_error = None
        last_error_time = None
        if self.error_history[uid]:
            last_err = self.error_history[uid][-1]
            last_error = last_err['error']
            last_error_time = last_err['time']
        
        # Determine status
        if now < sampler.pause_until:
            status = "paused"
            pause_reason = f"Chutes errors (until {time.strftime('%H:%M:%S', time.localtime(sampler.pause_until))})"
        else:
            status = "active"
            pause_reason = None
        
        # Calculate effective sampling rate (samples per day based on last hour)
        total_1h = sum(samples_1h.values())
        effective_daily_rate = total_1h * 24 if total_1h > 0 else 0.0
        
        return MinerSamplingStats(
            uid=uid,
            hotkey=sampler.miner.hotkey[:8] + "...",
            model=sampler.miner.model or "unknown",
            status=status,
            pause_until=sampler.pause_until,
            pause_reason=pause_reason,
            rate_multiplier=sampler.rate_multiplier,
            configured_daily_rate=sampler.config.daily_rate_per_env * len(sampler.envs),
            effective_daily_rate=effective_daily_rate,
            samples_1h=dict(samples_1h),
            total_samples_1h=total_1h,
            samples_24h=dict(samples_24h),
            total_samples_24h=sum(samples_24h.values()),
            consecutive_errors=sampler.consecutive_chutes_errors,
            last_error=last_error,
            last_error_time=last_error_time,
            error_count_1h=error_count_1h,
        )
    
    def _get_queue_stats(self) -> QueueStats:
        """Get queue statistics"""
        queue = self.scheduler.task_queue
        now = time.time()
        
        # Update throughput
        current_enqueued = queue.total_enqueued
        current_dequeued = queue.total_dequeued
        
        self.enqueue_history.append({
            'time': now,
            'count': current_enqueued - self.last_enqueue_count,
        })
        self.dequeue_history.append({
            'time': now,
            'count': current_dequeued - self.last_dequeue_count,
        })
        
        self.last_enqueue_count = current_enqueued
        self.last_dequeue_count = current_dequeued
        
        # Calculate rates (per minute)
        enqueue_rate = sum(e['count'] for e in self.enqueue_history) * 60 / len(self.enqueue_history) if self.enqueue_history else 0
        dequeue_rate = sum(d['count'] for d in self.dequeue_history) * 60 / len(self.dequeue_history) if self.dequeue_history else 0
        
        # Per-environment breakdown (approximate from pending batches)
        env_breakdown = defaultdict(int)
        for sampler_batches in queue.pending_batches.values():
            for env_batch in sampler_batches.values():
                if env_batch:
                    # Assume tasks in batch are for this env
                    env_name = env_batch[0].env_name if env_batch else "unknown"
                    env_breakdown[env_name] += len(env_batch)
        
        current_size = queue.qsize()
        
        return QueueStats(
            current_size=current_size,
            max_size=queue.max_size,
            warning_threshold=queue.warning_threshold,
            pause_threshold=queue.pause_threshold,
            resume_threshold=queue.resume_threshold,
            is_paused=queue.paused,
            is_warning=current_size >= queue.warning_threshold,
            enqueue_rate=enqueue_rate,
            dequeue_rate=dequeue_rate,
            total_enqueued=current_enqueued,
            total_dequeued=current_dequeued,
            env_breakdown=dict(env_breakdown),
        )
    
    def _get_worker_stats(self) -> WorkerStats:
        """Get worker statistics"""
        total_workers = len(self.scheduler.evaluation_workers)
        
        # Approximate active workers by semaphore
        # (total - available locked count)
        semaphore = self.scheduler.semaphore
        # Note: asyncio.Semaphore doesn't expose internal counter directly
        # We approximate based on task queue and completion times
        
        # Calculate average task duration
        avg_duration = 0.0
        if self.task_completion_times:
            avg_duration = sum(self.task_completion_times) / len(self.task_completion_times)
        
        # Calculate throughput (tasks per minute)
        tasks_per_minute = 0.0
        if self.dequeue_history:
            recent_dequeued = sum(d['count'] for d in list(self.dequeue_history)[-10:])
            tasks_per_minute = recent_dequeued * 6  # Last 10 seconds × 6 = per minute
        
        # Estimate active workers
        active_workers = min(
            int(tasks_per_minute * avg_duration / 60) if avg_duration > 0 else 0,
            total_workers
        )
        
        # Calculate utilization
        utilization = active_workers / total_workers if total_workers > 0 else 0.0
        
        return WorkerStats(
            total_workers=total_workers,
            active_workers=active_workers,
            idle_workers=total_workers - active_workers,
            tasks_per_minute=tasks_per_minute,
            avg_task_duration=avg_duration,
            utilization=utilization,
        )
    
    def get_status(self) -> SchedulerStatus:
        """Get complete scheduler status"""
        now = time.time()
        
        # Collect miner stats
        miner_stats = []
        for uid in self.scheduler.samplers.keys():
            stats = self._get_miner_stats(uid)
            if stats:
                miner_stats.append(stats)
        
        # Sort by UID
        miner_stats.sort(key=lambda m: m.uid)
        
        # Count active/paused miners
        active_miners = sum(1 for m in miner_stats if m.status == "active")
        paused_miners = sum(1 for m in miner_stats if m.status == "paused")
        
        # Get queue and worker stats
        queue_stats = self._get_queue_stats()
        worker_stats = self._get_worker_stats()
        
        # Calculate per-environment stats
        env_stats = defaultdict(lambda: {
            'total_samples_1h': 0,
            'total_samples_24h': 0,
            'queue_size': 0,
            'active_miners': 0,
        })
        
        for miner in miner_stats:
            for env_name, count in miner.samples_1h.items():
                env_stats[env_name]['total_samples_1h'] += count
            for env_name, count in miner.samples_24h.items():
                env_stats[env_name]['total_samples_24h'] += count
            if miner.status == "active":
                for env_name in miner.samples_1h.keys():
                    env_stats[env_name]['active_miners'] += 1
        
        for env_name, count in queue_stats.env_breakdown.items():
            env_stats[env_name]['queue_size'] = count
        
        return SchedulerStatus(
            timestamp=now,
            uptime_seconds=now - self.start_time,
            total_miners=len(miner_stats),
            active_miners=active_miners,
            paused_miners=paused_miners,
            queue=queue_stats,
            workers=worker_stats,
            miners=miner_stats,
            env_stats=dict(env_stats),
        )
    
    def get_status_dict(self) -> dict:
        """Get status as dictionary (JSON-serializable)"""
        status = self.get_status()
        return asdict(status)