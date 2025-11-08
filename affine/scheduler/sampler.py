import time
import random
import asyncio
from typing import List, Dict, Optional, Any
from affine.models import Miner
from affine.tasks import BaseSDKEnv
from affine.scheduler.models import Task
from affine.scheduler.queue import TaskQueue
from affine.scheduler.config import SamplingConfig
from affine.setup import logger


class MinerSampler:
    """Independent sampler for a single miner"""
    
    CHUTES_ERROR_PATTERNS = [
        "Invalid API key",           # Miner API key misconfigured
        "No instances available",    # Chutes instance not started or unavailable
        "HTTP 503",                  # Chutes service unavailable
        "HTTP 500",                  # Chutes internal server error
        "HTTP 429",                  # Rate limit (too many requests)
        "HTTP 402",                  # Chute creator has insufficient balance
        "Error code: 429",           # OpenAI rate limit error
        "Error code: 402",           # OpenAI insufficient balance
        "Error code: 401",           # OpenAI auth failed (invalid API key)
        "Error code: 503",           # OpenAI service unavailable
        "Error code: 500",           # OpenAI internal error
        "CHUTES_API_KEY",            # Chutes API key env var missing
        "maximum capacity",          # Chutes reached max capacity
        "try again later",           # Service busy, retry suggested
        "zero balance",              # Chute creator has zero balance
    ]
    
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
        
        self.rate_multiplier = 1.0
        self.error_count = 0
        self.pause_until = 0
        self.consecutive_chutes_errors = 0
        
        self.last_sample_time: Dict[str, float] = {}
        self.total_samples_today = 0
    
    def init_from_data(self, init_data: Dict[str, Any]):
        """Initialize sampler from historical data.
        
        Args:
            init_data: Dictionary containing:
                - 'total_samples': int - Total samples from Summary (for rate multiplier)
                - 'env_last_sample_time': Dict[str, float] - Last sample time per env
        """
        # Set total samples (used by scheduler to determine rate multiplier)
        self.total_samples_today = init_data.get('total_samples', 0)
        
        # Set last sample times per environment (for state display)
        self.last_sample_time = init_data.get('env_last_sample_time', {}).copy()
        
        logger.debug(
            f"[MinerSampler U{self.uid}] Initialized from history: "
            f"total_samples={self.total_samples_today}, "
            f"envs_with_history={len(self.last_sample_time)}"
        )
    
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
        # Record error to monitor
        if self.monitor:
            self.monitor.record_error(self.uid, error)
        
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