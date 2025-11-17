import time
import hashlib
from typing import Optional
from affine.models import Miner
from affine.tasks import BaseSDKEnv
from affine.scheduler.models import Task
from affine.scheduler.config import SamplingConfig
from affine.scheduler.error_classifier import is_service_error
from affine.setup import logger


class MinerSampler:
    """Miner state container for error handling and availability tracking.
    
    In global sampling mode, the scheduler coordinates all task creation.
    This class only manages per-miner state (errors, pauses, availability).
    """
    
    def __init__(
        self,
        uid: int,
        miner: Miner,
        config: SamplingConfig,
    ):
        self.uid = uid
        self.miner = miner
        self.config = config
        
        # Error tracking
        self.pause_until = 0.0
        self.consecutive_chutes_errors = 0
    
    def is_available(self) -> bool:
        """Check if miner is available for sampling."""
        if self.miner.slug is None:
            return False
        if time.time() < self.pause_until:
            return False
        return True
    
    def create_task(self, env: BaseSDKEnv, task_id: int) -> Task:
        """Create a task for this miner.
        
        Args:
            env: Environment to sample
            task_id: Task ID (provided by scheduler)
            
        Returns:
            Task object ready for queue
        """
        seed = self._generate_deterministic_seed(env.env_name, task_id)
        return Task(
            uid=self.uid,
            miner=self.miner,
            env_name=env.env_name,
            task_id=task_id,
            seed=seed,
        )
    
    def _generate_deterministic_seed(self, env_name: str, task_id: int) -> int:
        """Generate deterministic seed from env_name and task_id."""
        seed_input = f"{env_name}:{task_id}"
        hash_bytes = hashlib.sha256(seed_input.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder='big')
    
    def handle_error(self, error: str):
        """Handle task execution error with exponential backoff."""
        if is_service_error(error):
            self.consecutive_chutes_errors += 1
            
            if self.consecutive_chutes_errors % self.config.max_consecutive_errors == 0:
                pause_level = (self.consecutive_chutes_errors // self.config.max_consecutive_errors) - 1
                pause_duration = min(
                    self.config.chutes_error_pause_seconds * (2 ** pause_level),
                    self.config.max_pause_seconds
                )
                self.pause_until = time.time() + pause_duration
                
                logger.warning(
                    f"[Miner U{self.uid}] Paused for {pause_duration}s "
                    f"(level {pause_level}, errors: {self.consecutive_chutes_errors})"
                )
        else:
            self.consecutive_chutes_errors = 0
    
    def reset_error_state(self):
        """Reset error counters after successful sampling."""
        if self.consecutive_chutes_errors > 0:
            logger.info(f"[Miner U{self.uid}] Reset error state (had {self.consecutive_chutes_errors} errors)")
        self.consecutive_chutes_errors = 0
        self.pause_until = 0