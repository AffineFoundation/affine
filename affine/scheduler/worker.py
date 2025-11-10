import time
import asyncio
import traceback
from typing import Dict, Optional
from affine.tasks import BaseSDKEnv
from affine.models import Result
from affine.scheduler.models import Task
from affine.scheduler.queue import TaskQueue
from affine.setup import logger


class EvaluationWorker:
    """Evaluation worker that executes sampling tasks"""
    
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
        worker_id: int,
        task_queue: TaskQueue,
        result_queue: asyncio.Queue,
        envs: list[BaseSDKEnv],
        semaphore: asyncio.Semaphore,
        samplers: Dict[int, 'MinerSampler'],
        monitor: Optional['SchedulerMonitor'] = None,
    ):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.envs: Dict[str, BaseSDKEnv] = {env.env_name: env for env in envs}
        self.semaphore = semaphore
        self.samplers = samplers
        self.monitor = monitor
    
    async def run(self):
        """Main worker loop"""
        while True:
            try:
                task = await self.task_queue.get()
                
                async with self.semaphore:
                    result = await self._execute_task(task)
                
                if result:
                    # Handle errors - notify sampler and monitor
                    if result.error:
                        # Notify sampler to handle error (for pause logic)
                        sampler = self.samplers.get(task.uid)
                        if sampler:
                            sampler.handle_error(result.error)
                        
                        # Record error to monitor (for statistics)
                        if self.monitor:
                            self.monitor.record_error(task.uid, result.error)
                    
                    if result.error is None:
                        await self.result_queue.put(result)
                        logger.debug(
                            f"[RESULT] U{result.miner.uid:>3d} │ "
                            f"{result.env:<20} │ "
                            f"{result.score:>6.4f} │ "
                            f"{result.latency_seconds:>6.3f}s"
                        )
                    else:
                        logger.debug(
                            f"[SKIP]   U{result.miner.uid:>3d} │ "
                            f"{result.env:<20} │ "
                            f"Failed: {result.error}"
                        )
            
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[Worker-{self.worker_id}] Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task) -> Result:
        """Execute single evaluation task"""
        try:
            env = self.envs.get(task.env_name)
            if not env:
                logger.warning(f"Unknown env: {task.env_name}")
                return None
            
            result = await env.evaluate(task.miner, seed=task.seed)
            return result
        
        except Exception as e:
            return Result(
                miner=task.miner,
                env=task.env_name,
                score=0.0,
                latency_seconds=0.0,
                success=False,
                error=str(e),
                extra={},
                timestamp=time.time()
            )
    
    @classmethod
    def _is_chutes_error(cls, error_msg: str) -> bool:
        """Detect Chutes service errors"""
        return any(p in error_msg for p in cls.CHUTES_ERROR_PATTERNS)