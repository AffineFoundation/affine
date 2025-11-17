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
from affine.setup import logger

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
    
    # Sampling rates (global sampling - all miners get same rate)
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
    
    # Utilization (0-1)
    utilization: float = 0.0


@dataclass
class EvaluationSummary:
    """Overall evaluation statistics summary"""
    # Time windows
    total_samples_1h: int = 0
    total_samples_24h: int = 0
    
    # Sampling rates
    avg_samples_per_hour: float = 0.0
    projected_daily_samples: float = 0.0
    
    # Per-environment totals
    samples_by_env_1h: Dict[str, int] = field(default_factory=dict)
    samples_by_env_24h: Dict[str, int] = field(default_factory=dict)
    
    # Per-environment averages
    avg_samples_per_env_1h: Dict[str, float] = field(default_factory=dict)
    avg_samples_per_env_24h: Dict[str, float] = field(default_factory=dict)
    
    # Miner participation
    total_participating_miners: int = 0
    active_sampling_miners: int = 0
    
    # Error statistics
    total_errors_1h: int = 0
    miners_with_errors: int = 0
    
    # Throughput metrics
    current_throughput_per_minute: float = 0.0


@dataclass
class SchedulerStatus:
    """Complete scheduler status"""
    timestamp: float
    uptime_seconds: float
    
    # Summary
    total_miners: int
    active_miners: int
    paused_miners: int
    
    # Overall evaluation summary
    evaluation_summary: EvaluationSummary
    
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
    
    def __init__(self):
        self.scheduler = None  # Will be set after scheduler is created
        self.start_time = time.time()
        
        # Time-windowed counters
        self.sample_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=3600))
        self.error_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=3600))
        
        # Throughput tracking
        self.enqueue_history = deque(maxlen=60)  # Last 60 seconds
        self.dequeue_history = deque(maxlen=60)
        self.last_enqueue_count = 0
        self.last_dequeue_count = 0
        
        # Initialization flag
        self._initialized = False
    
    def set_scheduler(self, scheduler):
        """Set scheduler reference after initialization"""
        self.scheduler = scheduler
    
    async def load_historical_samples(self, blocks: int = 7200):
        """Load recent sample history to initialize counters after restart.
        
        This ensures the 1h/24h counters are accurate even after restart.
        Called once during scheduler startup.
        
        Args:
            blocks: Number of recent blocks to load (default: 7200 ≈ 1 day)
        """
        if self._initialized:
            logger.warning("[Monitor] Already initialized, skipping historical load")
            return
        
        try:
            from affine.storage import dataset
            
            logger.info(f"[Monitor] Loading last {blocks} blocks to initialize sample counters")
            
            now = time.time()
            one_hour_ago = now - 3600
            one_day_ago = now - 86400
            
            count_1h = 0
            count_24h = 0
            total_count = 0
            
            async for result in dataset(tail=blocks, compact=True):
                total_count += 1
                uid = result.miner.uid
                env = result.env
                timestamp = result.timestamp
                
                # Add to history deques
                if timestamp >= one_hour_ago:
                    self.sample_history[uid].append({
                        'time': timestamp,
                        'env': env,
                    })
                    count_1h += 1
                elif timestamp >= one_day_ago:
                    self.sample_history[uid].append({
                        'time': timestamp,
                        'env': env,
                    })
                    count_24h += 1
            
            self._initialized = True
            logger.info(
                f"[Monitor] Loaded {total_count} samples: "
                f"{count_1h} in last 1h, {count_24h} in last 24h"
            )
        
        except Exception as e:
            logger.error(f"[Monitor] Failed to load historical samples: {e}")
            # Continue anyway, monitor will just start from 0
            self._initialized = True
    
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
    
    def _get_miner_stats(self, uid: int) -> MinerSamplingStats:
        """Get statistics for a single miner"""
        if not self.scheduler:
            return None
        
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
            # Calculate current pause level from consecutive errors
            # Level = (consecutive_errors // 3) - 1 (since we're already paused)
            pause_level = max(0, (sampler.consecutive_chutes_errors // 3) - 1) if sampler.consecutive_chutes_errors >= 3 else 0
            
            # Calculate remaining pause duration
            remaining_seconds = int(sampler.pause_until - now)
            remaining_minutes = remaining_seconds // 60
            remaining_hours = remaining_minutes // 60
            remaining_mins = remaining_minutes % 60
            
            if remaining_hours > 0:
                duration_str = f"{remaining_hours}h{remaining_mins}m"
            else:
                duration_str = f"{remaining_minutes}m"
            
            pause_reason = f"Chutes errors (level {pause_level}, wait {duration_str}, until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sampler.pause_until))})"
        else:
            status = "active"
            pause_reason = None
        
        # Calculate effective sampling rate (samples per day based on last hour)
        total_1h = sum(samples_1h.values())
        effective_daily_rate = total_1h * 24 if total_1h > 0 else 0.0
        
        # Calculate configured daily rate from scheduler's global environments
        # In global sampling mode, all miners have the same configured rate
        configured_daily_rate = 0
        if self.scheduler and self.scheduler.envs:
            configured_daily_rate = sum(env.daily_rate for env in self.scheduler.envs)
        
        return MinerSamplingStats(
            uid=uid,
            hotkey=sampler.miner.hotkey[:8] + "...",
            model=sampler.miner.model or "unknown",
            status=status,
            pause_until=sampler.pause_until,
            pause_reason=pause_reason,
            configured_daily_rate=int(configured_daily_rate),
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
        """Get queue statistics (aggregated across all environment queues)"""
        if not self.scheduler or not self.scheduler.env_queues:
            return QueueStats(
                current_size=0,
                max_size=0,
                warning_threshold=0,
                pause_threshold=0,
                resume_threshold=0,
            )
        
        now = time.time()
        
        # Aggregate stats across all environment queues
        total_current_size = 0
        total_max_size = 0
        total_enqueued = 0
        total_dequeued = 0
        any_paused = False
        any_warning = False
        env_breakdown = {}
        
        # Use first queue for threshold values (all queues use same config)
        first_queue = next(iter(self.scheduler.env_queues.values()))
        warning_threshold = first_queue.warning_threshold
        pause_threshold = first_queue.pause_threshold
        resume_threshold = first_queue.resume_threshold
        
        for env_name, queue in self.scheduler.env_queues.items():
            current_size = queue.qsize()
            total_current_size += current_size
            total_max_size += queue.max_size
            total_enqueued += queue.total_enqueued
            total_dequeued += queue.total_dequeued
            
            if queue.paused:
                any_paused = True
            if current_size >= queue.warning_threshold:
                any_warning = True
            
            # Per-environment breakdown
            env_breakdown[env_name] = current_size
        
        # Update throughput tracking
        self.enqueue_history.append({
            'time': now,
            'count': total_enqueued - self.last_enqueue_count,
        })
        self.dequeue_history.append({
            'time': now,
            'count': total_dequeued - self.last_dequeue_count,
        })
        
        self.last_enqueue_count = total_enqueued
        self.last_dequeue_count = total_dequeued
        
        # Calculate rates (per minute)
        enqueue_rate = sum(e['count'] for e in self.enqueue_history) * 60 / len(self.enqueue_history) if self.enqueue_history else 0
        dequeue_rate = sum(d['count'] for d in self.dequeue_history) * 60 / len(self.dequeue_history) if self.dequeue_history else 0
        
        return QueueStats(
            current_size=total_current_size,
            max_size=total_max_size,
            warning_threshold=warning_threshold,
            pause_threshold=pause_threshold,
            resume_threshold=resume_threshold,
            is_paused=any_paused,
            is_warning=any_warning,
            enqueue_rate=enqueue_rate,
            dequeue_rate=dequeue_rate,
            total_enqueued=total_enqueued,
            total_dequeued=total_dequeued,
            env_breakdown=env_breakdown,
        )
    
    def _get_worker_stats(self) -> WorkerStats:
        """Get worker statistics"""
        if not self.scheduler:
            return WorkerStats(
                total_workers=0,
                active_workers=0,
                idle_workers=0,
            )
        
        total_workers = len(self.scheduler.evaluation_workers)
        
        # Calculate throughput (tasks per minute)
        tasks_per_minute = 0.0
        if self.dequeue_history:
            recent_dequeued = sum(d['count'] for d in list(self.dequeue_history)[-10:])
            tasks_per_minute = recent_dequeued * 6  # Last 10 seconds × 6 = per minute
        
        # Estimate active workers (assume 50% utilization as baseline)
        # This is a rough estimate since we don't track individual worker states
        active_workers = max(1, total_workers // 2) if tasks_per_minute > 0 else 0
        
        # Calculate utilization based on queue depth
        utilization = min(1.0, tasks_per_minute / (total_workers * 2)) if total_workers > 0 else 0.0
        
        return WorkerStats(
            total_workers=total_workers,
            active_workers=active_workers,
            idle_workers=total_workers - active_workers,
            tasks_per_minute=tasks_per_minute,
            utilization=utilization,
        )
    
    def get_status(self) -> SchedulerStatus:
        """Get complete scheduler status"""
        now = time.time()
        
        # Collect miner stats
        miner_stats = []
        if self.scheduler:
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
        
        # Calculate overall evaluation summary
        total_samples_1h = 0
        total_samples_24h = 0
        samples_by_env_1h = defaultdict(int)
        samples_by_env_24h = defaultdict(int)
        total_errors_1h = 0
        miners_with_errors = 0
        active_sampling_miners = 0
        
        for miner in miner_stats:
            # Aggregate samples
            total_samples_1h += miner.total_samples_1h
            total_samples_24h += miner.total_samples_24h
            
            # Per-environment aggregation
            for env_name, count in miner.samples_1h.items():
                env_stats[env_name]['total_samples_1h'] += count
                samples_by_env_1h[env_name] += count
            for env_name, count in miner.samples_24h.items():
                env_stats[env_name]['total_samples_24h'] += count
                samples_by_env_24h[env_name] += count
            
            # Active miners per environment
            if miner.status == "active":
                if miner.total_samples_1h > 0:
                    active_sampling_miners += 1
                for env_name in miner.samples_1h.keys():
                    env_stats[env_name]['active_miners'] += 1
            
            # Error tracking
            total_errors_1h += miner.error_count_1h
            if miner.error_count_1h > 0:
                miners_with_errors += 1
        
        for env_name, count in queue_stats.env_breakdown.items():
            env_stats[env_name]['queue_size'] = count
        
        # Calculate per-environment averages
        avg_samples_per_env_1h = {}
        avg_samples_per_env_24h = {}
        
        # Count participating miners per environment for averaging
        env_miner_counts = defaultdict(int)
        for miner in miner_stats:
            for env_name in miner.samples_1h.keys():
                env_miner_counts[env_name] += 1
        
        for env_name, total_count in samples_by_env_1h.items():
            miner_count = env_miner_counts.get(env_name, 0)
            avg_samples_per_env_1h[env_name] = total_count / miner_count if miner_count > 0 else 0.0
        
        for env_name, total_count in samples_by_env_24h.items():
            miner_count = env_miner_counts.get(env_name, 0)
            avg_samples_per_env_24h[env_name] = total_count / miner_count if miner_count > 0 else 0.0
        
        # Build evaluation summary
        evaluation_summary = EvaluationSummary(
            total_samples_1h=total_samples_1h,
            total_samples_24h=total_samples_24h,
            avg_samples_per_hour=total_samples_1h,
            projected_daily_samples=total_samples_1h * 24,
            samples_by_env_1h=dict(samples_by_env_1h),
            samples_by_env_24h=dict(samples_by_env_24h),
            avg_samples_per_env_1h=avg_samples_per_env_1h,
            avg_samples_per_env_24h=avg_samples_per_env_24h,
            total_participating_miners=len(miner_stats),
            active_sampling_miners=active_sampling_miners,
            total_errors_1h=total_errors_1h,
            miners_with_errors=miners_with_errors,
            current_throughput_per_minute=worker_stats.tasks_per_minute,
        )
        
        return SchedulerStatus(
            timestamp=now,
            uptime_seconds=now - self.start_time,
            total_miners=len(miner_stats),
            active_miners=active_miners,
            paused_miners=paused_miners,
            evaluation_summary=evaluation_summary,
            queue=queue_stats,
            workers=worker_stats,
            miners=miner_stats,
            env_stats=dict(env_stats),
        )
    
    def get_status_dict(self) -> dict:
        """Get status as dictionary (JSON-serializable)"""
        status = self.get_status()
        return asdict(status)