import time
import asyncio
import traceback
from typing import Dict, List, Optional
import bittensor as bt
from affine.models import Miner, Result
from affine.tasks import BaseSDKEnv
from affine.miners import miners as get_miners
from affine.storage import sink, load_summary
from affine.utils.subtensor import get_subtensor
from affine.setup import NETUID, logger
from affine.scheduler.config import SamplingConfig
from affine.scheduler.models import Task, SchedulerMetrics
from affine.scheduler.queue import TaskQueue
from affine.scheduler.sampler import MinerSampler
from affine.scheduler.worker import EvaluationWorker
from affine.scheduler.monitor import SchedulerMonitor


class SamplingScheduler:
    """Main sampling scheduler that coordinates all components"""
    
    def __init__(self, config: SamplingConfig, wallet: bt.wallet, enable_monitoring: bool = False, monitor_port: int = 8765):
        self.config = config
        self.wallet = wallet
        self.enable_monitoring = enable_monitoring
        self.monitor_port = monitor_port
        
        # Per-environment queues (created in start())
        self.env_queues: Dict[str, TaskQueue] = {}
        self.result_queue = asyncio.Queue()
        
        self.samplers: Dict[int, MinerSampler] = {}
        self.sampler_tasks: Dict[int, asyncio.Task] = {}
        
        self.evaluation_workers: List[asyncio.Task] = []
        self.upload_worker = None
        self.batch_flusher = None
        self.monitor_task = None
        self.miner_refresh_task = None
        self.api_server = None
        
        self.metrics = SchedulerMetrics()
        
        # Rate tracking for monitor loop
        self.last_monitor_time: Optional[float] = None
        self.last_total_enqueued: int = 0
        self.last_total_dequeued: int = 0
        self.last_env_enqueued: Dict[str, int] = {}
        self.last_env_dequeued: Dict[str, int] = {}
        
        # Initialize monitoring
        self.scheduler_monitor: Optional[SchedulerMonitor] = None
        if enable_monitoring:
            self.scheduler_monitor = SchedulerMonitor()
            self.scheduler_monitor.set_scheduler(self)
    
    async def start(self, envs: List[BaseSDKEnv]):
        """Start all scheduler components"""
        logger.info(f"[Scheduler] Starting with {len(envs)} environments")
        
        # Initialize monitor with historical data (if monitoring enabled)
        if self.scheduler_monitor:
            await self.scheduler_monitor.load_historical_samples(blocks=7200)
        
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
        
        # Start monitoring API if enabled
        if self.enable_monitoring and self.scheduler_monitor:
            self.api_server = asyncio.create_task(self._start_monitoring_api())
        
        logger.info("[Scheduler] All components started")
    
    async def _miner_refresh_loop(self, envs: List[BaseSDKEnv]):
        """Periodically refresh miner list and manage samplers"""
        while True:
            try:
                subtensor = await get_subtensor()
                meta = await subtensor.metagraph(NETUID)
                miners_map = await get_miners(meta=meta)
                
                # Start new samplers or update existing ones if miner info changed
                for uid, miner in miners_map.items():
                    if uid not in self.samplers:
                        # New miner - start sampler
                        await self._start_sampler(uid, miner, envs)
                    else:
                        # Existing miner - check if model/revision changed
                        existing_sampler = self.samplers[uid]
                        existing_miner = existing_sampler.miner
                        
                        # Check if critical miner attributes changed
                        if (existing_miner.model != miner.model or
                            existing_miner.revision != miner.revision or
                            existing_miner.slug != miner.slug):
                            
                            logger.info(
                                f"[Scheduler] U{uid} miner info changed: "
                                f"model={existing_miner.model}->{miner.model}, "
                                f"revision={existing_miner.revision}->{miner.revision}, "
                                f"slug={existing_miner.slug}->{miner.slug} - restarting sampler"
                            )
                            
                            # Stop old sampler and start new one with updated info
                            await self._stop_sampler(uid)
                            await self._start_sampler(uid, miner, envs)

                # Stop removed samplers
                for uid in list(self.samplers.keys()):
                    if uid not in miners_map:
                        await self._stop_sampler(uid)
                
                # Adjust sampling rates based on 24h counts
                await self._adjust_sampling_rates()
                
                self.metrics.active_miners = len(self.samplers)
                self.metrics.paused_miners = sum(
                    1 for s in self.samplers.values()
                    if time.time() < s.pause_until
                )
            
            except Exception as e:
                logger.error(f"[Scheduler] Miner refresh error: {e}")
                traceback.print_exc()
            
            await asyncio.sleep(self.config.miner_refresh_interval)
    
    async def _start_sampler(self, uid: int, miner: Miner, envs: List[BaseSDKEnv]):
        """Start sampler for a single miner"""
        sampler = MinerSampler(uid, miner, envs, self.config, monitor=self.scheduler_monitor)
        self.samplers[uid] = sampler
        
        # Pass env_queues dict to sampler for routing
        task = asyncio.create_task(sampler.run(self.env_queues))
        self.sampler_tasks[uid] = task
        
        logger.info(f"[Scheduler] Started sampler for U{uid}")
    
    async def _stop_sampler(self, uid: int):
        """Stop sampler for a single miner"""
        if uid in self.sampler_tasks:
            self.sampler_tasks[uid].cancel()
            del self.sampler_tasks[uid]
        if uid in self.samplers:
            del self.samplers[uid]
        logger.info(f"[Scheduler] Stopped sampler for U{uid}")
    
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
    
    async def _adjust_sampling_rates(self):
        """Adjust sampling rates by loading Summary data (50k blocks).
        
        This method loads the latest Summary file and checks sample counts
        for each miner's environment. If a miner's specific environment has less
        than the threshold (default 200), that environment gets accelerated
        sampling (3x multiplier).
        
        Called every 30 minutes to dynamically adjust rates based on actual performance.
        """
        try:
            # Load latest Summary (covers ~50k blocks)
            summary = await load_summary()
            miners_data = summary.get('data', {}).get('miners', {})
            
            # Build per-UID per-environment sample counts
            uid_env_counts: Dict[int, Dict[str, int]] = {}
            for hotkey, miner_info in miners_data.items():
                uid = miner_info.get('uid')
                if uid is None:
                    continue
                
                environments = miner_info.get('environments', {})
                uid_env_counts[uid] = {
                    env_name: env_data.get('count', 0)
                    for env_name, env_data in environments.items()
                }
            
            # Adjust rate multiplier per environment for each active sampler
            adjusted_miners = 0
            total_accelerated_envs = 0
            
            for uid, sampler in self.samplers.items():
                env_counts = uid_env_counts.get(uid, {})
                accelerated_envs = []
                
                # Check each environment independently
                for env in sampler.envs:
                    env_name = env.env_name
                    sample_count = env_counts.get(env_name, 0)
                    
                    if sample_count < self.config.low_sample_threshold:
                        sampler.env_rate_multipliers[env_name] = self.config.low_sample_multiplier
                        accelerated_envs.append(env_name)
                    else:
                        sampler.env_rate_multipliers[env_name] = 1.0
                
                if accelerated_envs:
                    total_accelerated_envs += len(accelerated_envs)
                    logger.debug(
                        f"[Scheduler] U{uid} accelerated envs: {', '.join(accelerated_envs)} "
                        f"({self.config.low_sample_multiplier}x)"
                    )
                
                adjusted_miners += 1
            
            logger.info(
                f"[Scheduler] Adjusted sampling rates: {adjusted_miners} miners, "
                f"{total_accelerated_envs} environment instances accelerated ({self.config.low_sample_multiplier}x)"
            )
        
        except Exception as e:
            logger.error(f"[Scheduler] Failed to adjust sampling rates: {e}")
            traceback.print_exc()
    
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
                
                logger.info(
                    f"[STATUS] samplers={self.metrics.active_miners} "
                    f"(paused={self.metrics.paused_miners}) "
                    f"total_queue={total_queue_size}(↓{total_enqueue_delta} ↑{total_dequeue_delta} in {elapsed_seconds:.0f}s) "
                    f"({env_stats}) "
                    f"result_queue={result_queue_size} "
                    f"enqueued={total_enqueued} dequeued={total_dequeued}"
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
        
        # Cancel all samplers
        for task in self.sampler_tasks.values():
            task.cancel()
        
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
        if self.api_server:
            self.api_server.cancel()
        
        # Wait for all to complete
        all_tasks = (
            list(self.sampler_tasks.values()) +
            self.evaluation_workers +
            [self.upload_worker, self.batch_flusher, self.monitor_task, self.miner_refresh_task]
        )
        if self.api_server:
            all_tasks.append(self.api_server)
        
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        logger.info("[Scheduler] Stopped")