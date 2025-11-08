import time
import asyncio
import traceback
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
    """Main sampling scheduler that coordinates all components"""
    
    def __init__(self, config: SamplingConfig, wallet: bt.wallet, enable_monitoring: bool = False, monitor_port: int = 8765):
        self.config = config
        self.wallet = wallet
        self.enable_monitoring = enable_monitoring
        self.monitor_port = monitor_port
        
        self.task_queue = TaskQueue(
            max_size=config.queue_max_size,
            warning_threshold=config.queue_warning_threshold,
            pause_threshold=config.queue_pause_threshold,
            resume_threshold=config.queue_resume_threshold,
            batch_size=config.batch_size,
        )
        self.result_queue = asyncio.Queue()
        
        self.samplers: Dict[int, MinerSampler] = {}
        self.sampler_tasks: Dict[int, asyncio.Task] = {}
        self.sample_counters: Dict[int, int] = {}
        
        self.evaluation_workers: List[asyncio.Task] = []
        self.upload_worker = None
        self.batch_flusher = None
        self.monitor_task = None
        self.miner_refresh_task = None
        self.api_server = None
        
        self.metrics = SchedulerMetrics()
        self.semaphore = asyncio.Semaphore(config.num_evaluation_workers)
        
        # Initialize monitoring
        self.scheduler_monitor: Optional[SchedulerMonitor] = None
        if enable_monitoring:
            self.scheduler_monitor = SchedulerMonitor()
            self.scheduler_monitor.set_scheduler(self)
    
    async def start(self, envs: List[BaseSDKEnv]):
        """Start all scheduler components"""
        logger.info(f"[Scheduler] Starting with {len(envs)} environments")
        
        # Load historical data for initialization
        from affine.scheduler.initializer import SchedulerInitializer
        subtensor = await get_subtensor()
        current_block = await subtensor.get_current_block()
        
        initializer = SchedulerInitializer()
        init_data = await initializer.load_init_data(current_block)
        logger.info(f"[Scheduler] Loaded initialization data for {len(init_data)} miners")
        
        # Store init data for later use in miner refresh
        self._init_data = init_data
        
        # Start evaluation workers
        for i in range(self.config.num_evaluation_workers):
            worker = EvaluationWorker(
                worker_id=i,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                envs=envs,
                semaphore=self.semaphore,
                monitor=self.scheduler_monitor,
            )
            task = asyncio.create_task(worker.run())
            self.evaluation_workers.append(task)
        
        logger.info(f"[Scheduler] Started {len(self.evaluation_workers)} evaluation workers")
        
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
                
                # Start new samplers
                for uid, miner in miners_map.items():
                    if uid not in self.samplers:
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
        
        # Initialize from historical data if available
        if hasattr(self, '_init_data') and uid in self._init_data:
            sampler.init_from_data(self._init_data[uid])
        
        self.samplers[uid] = sampler
        
        task = asyncio.create_task(sampler.run(self.task_queue))
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
    
    async def _adjust_sampling_rates(self):
        """Adjust sampling rates based on 24h sample counts"""
        for uid, sampler in self.samplers.items():
            total_samples = self.sample_counters.get(uid, 0)
            
            if total_samples < self.config.low_sample_threshold:
                sampler.rate_multiplier = self.config.low_sample_multiplier
            else:
                sampler.rate_multiplier = 1.0
    
    async def _batch_flusher_loop(self):
        """Periodically flush incomplete batches"""
        while True:
            try:
                await asyncio.sleep(self.config.batch_flush_interval)
                await self.task_queue.flush_all_batches()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[Scheduler] Batch flusher error: {e}")
    
    async def _monitor_loop(self):
        """Monitor and log status"""
        while True:
            try:
                await asyncio.sleep(30)
                
                queue_size = self.task_queue.qsize()
                result_queue_size = self.result_queue.qsize()
                
                logger.info(
                    f"[STATUS] samplers={self.metrics.active_miners} "
                    f"(paused={self.metrics.paused_miners}) "
                    f"task_queue={queue_size} result_queue={result_queue_size} "
                    f"enqueued={self.task_queue.total_enqueued} "
                    f"dequeued={self.task_queue.total_dequeued}"
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
                    
                    # Update sample counter
                    if result.miner and result.miner.uid:
                        self.sample_counters[result.miner.uid] = \
                            self.sample_counters.get(result.miner.uid, 0) + 1
                else:
                    try:
                        result = await asyncio.wait_for(
                            self.result_queue.get(),
                            timeout=5.0
                        )
                        batch.append(result)
                        
                        if result.miner and result.miner.uid:
                            self.sample_counters[result.miner.uid] = \
                                self.sample_counters.get(result.miner.uid, 0) + 1
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
                self.task_queue,
                self.config.num_evaluation_workers,
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