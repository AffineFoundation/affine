import time
import random
import asyncio
from typing import List, Dict, Optional, Any
from affine.models import Miner
from affine.tasks import BaseSDKEnv
from affine.scheduler.models import Task
from affine.scheduler.queue import TaskQueue
from affine.scheduler.config import SamplingConfig
from affine.scheduler.error_classifier import is_service_error
from affine.setup import logger


class MinerSampler:
    """Independent sampler for a single miner"""
    
    def __init__(
        self,
        uid: int,
        miner: Miner,
        envs: List[BaseSDKEnv],
        config: SamplingConfig,
        monitor: Optional['SchedulerMonitor'] = None,
    ):
        self.uid = uid
        self.miner = miner
        self.envs = envs
        self.config = config
        self.monitor = monitor
        
        # Per-environment rate multipliers (env_name -> multiplier)
        self.env_rate_multipliers: Dict[str, float] = {env.env_name: 1.0 for env in envs}
        self.error_count = 0
        self.pause_until = 0
        self.consecutive_chutes_errors = 0
        
        self.last_sample_time: Dict[str, float] = {}
    
    async def run(self, env_queues: Dict[str, TaskQueue]):
        """Main sampling loop
        
        Args:
            env_queues: Dict mapping env_name to TaskQueue for that environment
        """
        while True:
            try:
                if time.time() < self.pause_until:
                    await asyncio.sleep(1)
                    continue
                
                if not self._check_miner_available():
                    await asyncio.sleep(60)
                    continue
                
                for env in self.envs:
                    if self._should_sample(env):
                        task = self._create_task(env)
                        
                        # Route task to the appropriate environment queue
                        queue = env_queues.get(env.env_name)
                        if queue:
                            await queue.put(task, sampler_id=self.uid)
                            self._update_sample_time(env)
                            
                            # Record sampling event to monitor
                            if self.monitor:
                                self.monitor.record_sample(self.uid, env.env_name)
                
                await asyncio.sleep(self._next_check_interval())
            
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[MinerSampler U{self.uid}] Error: {e}")
                await asyncio.sleep(5)
    
    def _should_sample(self, env: BaseSDKEnv) -> bool:
        """Check if should sample this env based on its configured rate"""
        env_daily_rate = env.daily_rate
        # Use per-environment multiplier
        multiplier = self.env_rate_multipliers.get(env.env_name, 1.0)
        interval = 86400 / (env_daily_rate * multiplier)

        current_time = time.time()
        last_time = self.last_sample_time.get(env.env_name)

        if last_time is None:
            random_offset = random.uniform(0, interval)
            self.last_sample_time[env.env_name] = current_time - random_offset
            last_time = self.last_sample_time[env.env_name]

        return (current_time - last_time) >= interval

    def _create_task(self, env: BaseSDKEnv) -> Task:
        """Create sampling task for env"""
        return Task(
            uid=self.uid,
            miner=self.miner,
            env_name=env.env_name,
            seed=random.randint(0, 2**32 - 1),
        )
    
    def _update_sample_time(self, env: BaseSDKEnv):
        """Record sample time"""
        self.last_sample_time[env.env_name] = time.time()
    
    def _check_miner_available(self) -> bool:
        """Check if miner is available"""
        return self.miner.slug is not None
    
    def _next_check_interval(self) -> float:
        """Calculate next check interval based on fastest environment
        
        Returns the interval at which to check if any environment needs sampling.
        This should be significantly shorter than the fastest sampling interval
        to ensure timely sampling, but not so short as to waste CPU.
        """
        if not self.envs:
            return 60.0
        
        # Find the minimum sampling interval across all environments
        # (highest daily rate with multiplier = shortest interval)
        min_interval = float('inf')
        for env in self.envs:
            multiplier = self.env_rate_multipliers.get(env.env_name, 1.0)
            interval = 86400 / (env.daily_rate * multiplier)
            min_interval = min(min_interval, interval)
        
        # Check at half the minimum interval to ensure timely sampling
        # But cap at reasonable bounds: min 1s, max 60s
        check_interval = min_interval / 2
        return max(1.0, min(check_interval, 60.0))
    
    def handle_error(self, error: str):
        """Handle task execution error with exponential backoff
        
        Uses centralized error classifier to determine if this is a service error
        that should trigger sampling pause. Implements exponential backoff:
        - 3 errors (level 0): 10 minutes (600s)
        - 6 errors (level 1): 20 minutes (1200s)
        - 9 errors (level 2): 40 minutes (2400s)
        - Max pause: 24 hours (86400s)
        
        Pause level is calculated as: consecutive_chutes_errors // 3
        """
        # Use centralized error classifier
        if is_service_error(error):
            self.consecutive_chutes_errors += 1
            
            # Trigger pause every 3 errors
            if self.consecutive_chutes_errors % self.config.max_consecutive_errors == 0:
                # Calculate pause level and duration
                pause_level = (self.consecutive_chutes_errors // self.config.max_consecutive_errors) - 1
                base_pause = self.config.chutes_error_pause_seconds
                pause_duration = min(
                    base_pause * (2 ** pause_level),
                    self.config.max_pause_seconds
                )
                
                self.pause_until = time.time() + pause_duration
                
                logger.warning(
                    f"[MinerSampler U{self.uid}] Paused for {pause_duration}s "
                    f"(level {pause_level}, total errors: {self.consecutive_chutes_errors}) due to Chutes service errors"
                )
        else:
            # Non-service error - reset counter
            self.consecutive_chutes_errors = 0
    
    def reset_error_state(self):
        """Reset error counters after successful sampling
        
        Should be called when a sample completes successfully (not a service error).
        This allows miners to recover from temporary service issues.
        """
        if self.consecutive_chutes_errors > 0:
            pause_level = max(0, (self.consecutive_chutes_errors // self.config.max_consecutive_errors) - 1)
            logger.info(
                f"[MinerSampler U{self.uid}] Successful sample - resetting error state "
                f"(had {self.consecutive_chutes_errors} errors, was at level {pause_level})"
            )
        self.consecutive_chutes_errors = 0
        self.pause_until = 0