import time
import json
import asyncio
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import bittensor as bt
from affine.models import Miner, Result
from affine.tasks import BaseSDKEnv
from affine.miners import miners as get_miners
from affine.storage import sink
from affine.utils.subtensor import get_subtensor
from affine.setup import NETUID, logger
from affine.scheduler.config import SamplingConfig
from affine.scheduler.models import Task, SchedulerMetrics
from affine.scheduler.queue import TaskQueue
from affine.scheduler.sampler import MinerSampler
from affine.scheduler.worker import EvaluationWorker
from affine.scheduler.monitor import SchedulerMonitor


class SamplingScheduler:
    """Main sampling scheduler that coordinates all components.
    
    Uses environment-based global sequential sampling where:
    - Each environment is sampled at its daily_rate (default = dataset size)
    - All miners receive the same task_id for each sampling event
    - Task IDs progress sequentially through the dataset range
    """
    
    def __init__(self, config: SamplingConfig, wallet: bt.wallet, enable_monitoring: bool = False, monitor_port: int = 8765):
        self.config = config
        self.wallet = wallet
        self.enable_monitoring = enable_monitoring
        self.monitor_port = monitor_port
        
        # Per-environment queues (created in start())
        self.env_queues: Dict[str, TaskQueue] = {}
        self.result_queue = asyncio.Queue()
        
        # Miner state (simplified - no longer runs independent sampling loops)
        self.samplers: Dict[int, MinerSampler] = {}
        
        self.evaluation_workers: List[asyncio.Task] = []
        self.upload_worker = None
        self.batch_flusher = None
        self.monitor_task = None
        self.miner_refresh_task = None
        self.api_server = None
        self.global_sampling_task = None
        
        self.metrics = SchedulerMetrics()
        
        # Rate tracking for monitor loop
        self.last_monitor_time: Optional[float] = None
        self.last_env_enqueued: Dict[str, int] = {}
        self.last_env_dequeued: Dict[str, int] = {}
        
        # Initialize monitoring
        self.scheduler_monitor: Optional[SchedulerMonitor] = None
        if enable_monitoring:
            self.scheduler_monitor = SchedulerMonitor()
            self.scheduler_monitor.set_scheduler(self)
        
        # Environments list (set in start())
        self.envs: List[BaseSDKEnv] = []
        
        # Global sampling state (per-environment)
        self.env_task_counters: Dict[str, int] = {}  # Current task_id for each env
        self.last_global_sample_time: Dict[str, float] = {}  # Last sample time for rate control
        
        # State persistence
        self.state_file = Path.home() / ".cache" / "affine" / "sampling_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def start(self, envs: List[BaseSDKEnv]):
        """Start all scheduler components"""
        logger.info(f"[Scheduler] Starting with {len(envs)} environments")
        
        # Store environments for global sampling
        self.envs = envs
        
        # Initialize monitor with historical data (if monitoring enabled)
        if self.scheduler_monitor:
            await self.scheduler_monitor.load_historical_samples(blocks=7200)
        
        # Load persisted state or initialize
        self._load_state()
        
        # Initialize task counters for each environment
        for env in envs:
            env_name = env.env_name
            start_index = env.start_index
            end_index = env.end_index
            
            # Initialize counter if not already set
            if env_name not in self.env_task_counters:
                self.env_task_counters[env_name] = start_index
            
            # Ensure counter is within valid range
            if self.env_task_counters[env_name] >= end_index:
                self.env_task_counters[env_name] = start_index  # Wrap around
            
            logger.info(
                f"[Scheduler] {env_name}: range=[{start_index}, {end_index}), "
                f"daily_rate={env.daily_rate}, next_task_id={self.env_task_counters[env_name]}"
            )
        
        # Create per-environment queues
        for env in envs:
            self.env_queues[env.env_name] = TaskQueue(
                max_size=self.config.queue_max_size_per_env,
                warning_threshold=self.config.queue_warning_threshold,
                pause_threshold=self.config.queue_pause_threshold,
                resume_threshold=self.config.queue_resume_threshold,
                batch_size=self.config.batch_size,
            )
            logger.debug(f"[Scheduler] Created queue for {env.env_name} (max_size={self.config.queue_max_size_per_env})")
        
        # Start per-environment workers
        worker_count = 0
        for env in envs:
            env_queue = self.env_queues[env.env_name]
            for i in range(self.config.workers_per_env):
                worker = EvaluationWorker(
                    worker_id=f"{env.env_name}-{i}",
                    task_queue=env_queue,
                    result_queue=self.result_queue,
                    env=env,
                    samplers=self.samplers,
                    monitor=self.scheduler_monitor,
                )
                task = asyncio.create_task(worker.run())
                self.evaluation_workers.append(task)
                worker_count += 1
        
        logger.info(f"[Scheduler] Started {worker_count} evaluation workers ({self.config.workers_per_env} per environment)")
        
        # Start upload worker
        self.upload_worker = asyncio.create_task(self._upload_worker_loop())
        
        # Start batch flusher
        self.batch_flusher = asyncio.create_task(self._batch_flusher_loop())
        
        # Start monitor
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        # Start miner refresh
        self.miner_refresh_task = asyncio.create_task(self._miner_refresh_loop(envs))
        
        # Start global sampling coordinator
        self.global_sampling_task = asyncio.create_task(self._global_sampling_loop())
        
        # Start monitoring API if enabled
        if self.enable_monitoring and self.scheduler_monitor:
            self.api_server = asyncio.create_task(self._start_monitoring_api())
        
        logger.info("[Scheduler] All components started")
    
    def _load_state(self):
        """Load persisted sampling state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.env_task_counters = state.get('env_task_counters', {})
                logger.info(f"[Scheduler] Loaded state from {self.state_file}")
            except Exception as e:
                logger.warning(f"[Scheduler] Failed to load state: {e}")
                self.env_task_counters = {}
    
    def _save_state(self):
        """Save sampling state to file for persistence."""
        try:
            state = {'env_task_counters': self.env_task_counters}
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"[Scheduler] Failed to save state: {e}")
    
    async def _miner_refresh_loop(self, envs: List[BaseSDKEnv]):
        """Periodically refresh miner list"""
        while True:
            try:
                subtensor = await get_subtensor()
                meta = await subtensor.metagraph(NETUID)
                miners_map = await get_miners(meta=meta)
                
                # Add new miners or update existing ones
                for uid, miner in miners_map.items():
                    if uid not in self.samplers:
                        self.samplers[uid] = MinerSampler(uid, miner, self.config)
                        logger.info(f"[Scheduler] Added miner U{uid}")
                    else:
                        # Update miner info if changed
                        existing = self.samplers[uid].miner
                        if (existing.hotkey != miner.hotkey or
                            existing.model != miner.model or
                            existing.slug != miner.slug):
                            self.samplers[uid] = MinerSampler(uid, miner, self.config)
                            logger.info(f"[Scheduler] Updated miner U{uid}")
                
                # Remove miners no longer in metagraph
                for uid in list(self.samplers.keys()):
                    if uid not in miners_map:
                        del self.samplers[uid]
                        logger.info(f"[Scheduler] Removed miner U{uid}")
                
                self.metrics.active_miners = len(self.samplers)
                self.metrics.paused_miners = sum(
                    1 for s in self.samplers.values() if not s.is_available()
                )
            
            except Exception as e:
                logger.error(f"[Scheduler] Miner refresh error: {e}")
                traceback.print_exc()
            
            await asyncio.sleep(self.config.miner_refresh_interval)
    
    async def _global_sampling_loop(self):
        """Global sampling coordinator - creates tasks for all miners with the same task_id.
        
        Rate-controlled by each environment's daily_rate:
        - interval = 86400 / daily_rate seconds between samples
        - All miners get the same task_id for consistent evaluation
        """
        logger.info("[Scheduler] Global sampling started")
        
        # Initialize with staggered offsets
        import random
        for env in self.envs:
            interval = 86400.0 / env.daily_rate
            self.last_global_sample_time[env.env_name] = time.time() - random.uniform(0, interval)
        
        while True:
            try:
                if not self.samplers or not self.envs:
                    await asyncio.sleep(5)
                    continue
                
                current_time = time.time()
                
                # Find most overdue environment
                best_env = None
                max_overdue = 0.0
                
                for env in self.envs:
                    interval = 86400.0 / env.daily_rate
                    elapsed = current_time - self.last_global_sample_time.get(env.env_name, 0)
                    overdue = elapsed / interval
                    
                    if overdue >= 1.0 and overdue > max_overdue:
                        queue = self.env_queues.get(env.env_name)
                        if queue and queue.qsize() < self.config.queue_max_size_per_env * 0.8:
                            max_overdue = overdue
                            best_env = env
                
                if best_env:
                    env = best_env
                    queue = self.env_queues[env.env_name]
                    
                    # Get next task_id and increment counter
                    task_id = self.env_task_counters[env.env_name]
                    self.env_task_counters[env.env_name] += 1
                    
                    # Wrap around if reached end
                    if self.env_task_counters[env.env_name] >= env.end_index:
                        self.env_task_counters[env.env_name] = env.start_index
                    
                    # Create tasks for all available miners
                    tasks_created = 0
                    for uid, sampler in self.samplers.items():
                        if sampler.is_available():
                            task = sampler.create_task(env, task_id)
                            await queue.put(task, sampler_id=uid)
                            tasks_created += 1
                            
                            if self.scheduler_monitor:
                                self.scheduler_monitor.record_sample(uid, env.env_name)
                    
                    self.last_global_sample_time[env.env_name] = current_time
                    
                    if tasks_created > 0:
                        logger.debug(f"[Sampling] {env.env_name}: task_id={task_id}, miners={tasks_created}")
                    
                    self._save_state()
                    
                    await asyncio.sleep(0.01)
                else:
                    # Calculate sleep until next sample
                    min_sleep = 60.0
                    for env in self.envs:
                        interval = 86400.0 / env.daily_rate
                        elapsed = current_time - self.last_global_sample_time.get(env.env_name, 0)
                        time_until_next = max(0.1, interval - elapsed)
                        min_sleep = min(min_sleep, time_until_next)
                    
                    await asyncio.sleep(min_sleep)
            
            except asyncio.CancelledError:
                self._save_state()  # Save on shutdown
                raise
            except Exception as e:
                logger.error(f"[Scheduler] Sampling error: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)
    
    def _calculate_deltas(self, current_time: float) -> tuple[Dict[str, int], Dict[str, int], int, int, float]:
        """Calculate enqueue/dequeue deltas since last check.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Tuple of (env_enqueue_deltas, env_dequeue_deltas, total_enqueue_delta, total_dequeue_delta, elapsed_seconds)
        """
        env_enqueue_deltas = {}
        env_dequeue_deltas = {}
        
        if self.last_monitor_time is None:
            return env_enqueue_deltas, env_dequeue_deltas, 0, 0, 0.0
        
        elapsed_seconds = current_time - self.last_monitor_time
        if elapsed_seconds <= 0:
            return env_enqueue_deltas, env_dequeue_deltas, 0, 0, 0.0
        
        # Calculate deltas
        total_enqueue_delta = 0
        total_dequeue_delta = 0
        
        for name, q in self.env_queues.items():
            env_enqueue = q.total_enqueued - self.last_env_enqueued.get(name, 0)
            env_dequeue = q.total_dequeued - self.last_env_dequeued.get(name, 0)
            
            env_enqueue_deltas[name] = env_enqueue
            env_dequeue_deltas[name] = env_dequeue
            
            total_enqueue_delta += env_enqueue
            total_dequeue_delta += env_dequeue
        
        return env_enqueue_deltas, env_dequeue_deltas, total_enqueue_delta, total_dequeue_delta, elapsed_seconds
    
    def _update_rate_tracking(self, current_time: float):
        """Update tracking variables for delta calculation.
        
        Args:
            current_time: Current timestamp
        """
        self.last_monitor_time = current_time
        
        for name, q in self.env_queues.items():
            self.last_env_enqueued[name] = q.total_enqueued
            self.last_env_dequeued[name] = q.total_dequeued
    
    
    async def _batch_flusher_loop(self):
        """Periodically flush incomplete batches"""
        while True:
            try:
                await asyncio.sleep(self.config.batch_flush_interval)
                # Flush batches for all environment queues
                for env_queue in self.env_queues.values():
                    await env_queue.flush_all_batches()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[Scheduler] Batch flusher error: {e}")
    
    async def _monitor_loop(self):
        """Monitor and log status"""
        while True:
            try:
                await asyncio.sleep(30)
                
                current_time = time.time()
                
                # Aggregate queue stats across all environments
                total_queue_size = sum(q.qsize() for q in self.env_queues.values())
                total_enqueued = sum(q.total_enqueued for q in self.env_queues.values())
                total_dequeued = sum(q.total_dequeued for q in self.env_queues.values())
                result_queue_size = self.result_queue.qsize()
                
                # Initialize tracking on first run
                if self.last_monitor_time is None:
                    self._update_rate_tracking(current_time)
                    logger.info(
                        f"[STATUS] samplers={self.metrics.active_miners} "
                        f"(paused={self.metrics.paused_miners}) "
                        f"total_queue={total_queue_size} "
                        f"result_queue={result_queue_size} "
                        f"enqueued={total_enqueued} dequeued={total_dequeued} "
                        f"(tracking initialized)"
                    )
                    continue
                
                # Calculate deltas using helper method
                env_enqueue_deltas, env_dequeue_deltas, total_enqueue_delta, total_dequeue_delta, elapsed_seconds = \
                    self._calculate_deltas(current_time)
                
                # Update tracking for next iteration
                self._update_rate_tracking(current_time)
                
                # Format per-environment breakdown: queue_size (↓input_delta ↑output_delta)
                env_stats = " ".join([
                    f"{name}={q.qsize()}(↓{env_enqueue_deltas.get(name, 0)} ↑{env_dequeue_deltas.get(name, 0)})"
                    for name, q in self.env_queues.items()
                ])
                
                # Add global sampling progress
                global_progress = ""
                progress_info = []
                for env in self.envs:
                    task_id = self.env_task_counters.get(env.env_name, env.start_index)
                    total = env.end_index - env.start_index
                    done = task_id - env.start_index
                    pct = (done / total * 100) if total > 0 else 0
                    progress_info.append(f"{env.env_name.split(':')[-1]}@{task_id}({pct:.0f}%)")
                if progress_info:
                    global_progress = f" [{' '.join(progress_info)}]"
                
                logger.info(
                    f"[STATUS] samplers={self.metrics.active_miners} "
                    f"(paused={self.metrics.paused_miners}) "
                    f"total_queue={total_queue_size}(↓{total_enqueue_delta} ↑{total_dequeue_delta} in {elapsed_seconds:.0f}s) "
                    f"({env_stats}) "
                    f"result_queue={result_queue_size} "
                    f"enqueued={total_enqueued} dequeued={total_dequeued}"
                    f"{global_progress}"
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[Scheduler] Monitor error: {e}")
    
    async def _upload_worker_loop(self):
        """Upload results in batches"""
        batch: List[Result] = []
        batch_start_time = None
        
        while True:
            try:
                if not batch:
                    result = await self.result_queue.get()
                    batch.append(result)
                    batch_start_time = time.monotonic()
                else:
                    try:
                        result = await asyncio.wait_for(
                            self.result_queue.get(),
                            timeout=5.0
                        )
                        batch.append(result)
                    except asyncio.TimeoutError:
                        pass
                
                elapsed = time.monotonic() - batch_start_time if batch_start_time else 0
                
                if len(batch) >= self.config.sink_batch_size or \
                   (batch and elapsed >= self.config.sink_max_wait):
                    await self._upload_batch(batch)
                    batch.clear()
                    batch_start_time = None
            
            except asyncio.CancelledError:
                if batch:
                    await self._upload_batch(batch)
                raise
            except Exception as e:
                logger.error(f"[Scheduler] Upload worker error: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
    
    async def _start_monitoring_api(self):
        """Start monitoring API server"""
        try:
            from affine.scheduler.api import start_monitoring_server
            
            logger.info(f"[Scheduler] Starting monitoring API on port {self.monitor_port}")
            await start_monitoring_server(
                self.scheduler_monitor,
                port=self.monitor_port
            )
        except Exception as e:
            logger.error(f"[Scheduler] Failed to start monitoring API: {e}")
            traceback.print_exc()
    
    async def _upload_batch(self, batch: List[Result]):
        """Upload batch of results to storage"""
        try:
            subtensor = await get_subtensor()
            block = await subtensor.get_current_block()
            
            await sink(self.wallet, batch, block)
            
            self.metrics.total_results_uploaded += len(batch)
            logger.debug(f"[Scheduler] Uploaded {len(batch)} results to storage")
        
        except Exception as e:
            logger.error(f"[Scheduler] Upload failed: {e}")
            traceback.print_exc()
    
    async def stop(self):
        """Stop all scheduler components"""
        logger.info("[Scheduler] Stopping...")
        
        # Save state before stopping
        self._save_state()
        
        # Cancel workers
        for worker in self.evaluation_workers:
            worker.cancel()
        
        # Cancel other tasks
        if self.upload_worker:
            self.upload_worker.cancel()
        if self.batch_flusher:
            self.batch_flusher.cancel()
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.miner_refresh_task:
            self.miner_refresh_task.cancel()
        if self.global_sampling_task:
            self.global_sampling_task.cancel()
        if self.api_server:
            self.api_server.cancel()
        
        # Wait for all to complete
        all_tasks = self.evaluation_workers + [
            self.upload_worker, self.batch_flusher,
            self.monitor_task, self.miner_refresh_task
        ]
        if self.global_sampling_task:
            all_tasks.append(self.global_sampling_task)
        if self.api_server:
            all_tasks.append(self.api_server)
        
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        logger.info("[Scheduler] Stopped")