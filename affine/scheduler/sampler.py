import time
import random
import asyncio
from typing import List, Dict
from affine.models import Miner
from affine.tasks import BaseSDKEnv
from affine.scheduler.models import Task
from affine.scheduler.queue import TaskQueue
from affine.scheduler.config import SamplingConfig
from affine.setup import logger


class MinerSampler:
    """Independent sampler for a single miner"""
    
    CHUTES_ERROR_PATTERNS = [
        "Invalid API key",
        "No instances available",
        "HTTP 503",
        "HTTP 500",
        "HTTP 429",
        "Error code: 429",
        "RateLimitError",
        "Error code: 401",
        "Error code: 503",
        "Error code: 500",
        "CHUTES_API_KEY",
        "maximum capacity",
        "try again later",
    ]
    
    def __init__(
        self,
        uid: int,
        miner: Miner,
        envs: List[BaseSDKEnv],
        config: SamplingConfig,
    ):
        self.uid = uid
        self.miner = miner
        self.envs = envs
        self.config = config
        
        self.rate_multiplier = 1.0
        self.error_count = 0
        self.pause_until = 0
        self.consecutive_chutes_errors = 0
        
        self.last_sample_time: Dict[str, float] = {}
        self.total_samples_today = 0
    
    async def run(self, task_queue: TaskQueue):
        """Main sampling loop"""
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
                        await task_queue.put(task, sampler_id=self.uid)
                        self._update_sample_time(env)
                
                await asyncio.sleep(self._next_check_interval())
            
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[MinerSampler U{self.uid}] Error: {e}")
                await asyncio.sleep(5)
    
    def _should_sample(self, env: BaseSDKEnv) -> bool:
        """Check if should sample this env based on its configured rate"""
        # Use environment-specific daily rate
        env_daily_rate = env.daily_rate
        interval = 86400 / (env_daily_rate * self.rate_multiplier)
        last_time = self.last_sample_time.get(env.env_name, 0)
        return (time.time() - last_time) >= interval
    
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
        self.total_samples_today += 1
    
    def _check_miner_available(self) -> bool:
        """Check if miner is available"""
        return self.miner.slug is not None
    
    def _next_check_interval(self) -> float:
        """Calculate next check interval based on fastest environment"""
        if not self.envs:
            return 60.0
        
        # Use the highest daily rate among all environments to determine check frequency
        max_daily_rate = max(env.daily_rate for env in self.envs)
        base_interval = 86400 / (max_daily_rate * self.rate_multiplier * len(self.envs))
        return max(1.0, min(base_interval / 2, 60.0))
    
    def handle_error(self, error: str):
        """Handle task execution error"""
        if self._is_chutes_error(error):
            self.consecutive_chutes_errors += 1
            
            if self.consecutive_chutes_errors >= self.config.max_consecutive_errors:
                self.pause_until = time.time() + self.config.chutes_error_pause_seconds
                logger.warning(
                    f"[MinerSampler U{self.uid}] Paused for "
                    f"{self.config.chutes_error_pause_seconds}s due to Chutes errors"
                )
        else:
            self.consecutive_chutes_errors = 0
    
    @classmethod
    def _is_chutes_error(cls, error_msg: str) -> bool:
        """Detect Chutes service errors"""
        return any(pattern in error_msg for pattern in cls.CHUTES_ERROR_PATTERNS)