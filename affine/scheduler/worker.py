import time
import asyncio
import traceback
from typing import Dict
from affine.tasks import BaseSDKEnv
from affine.models import Result
from affine.scheduler.models import Task
from affine.scheduler.queue import TaskQueue
from affine.setup import logger


class EvaluationWorker:
    """Evaluation worker that executes sampling tasks"""
    
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
        worker_id: int,
        task_queue: TaskQueue,
        result_queue: asyncio.Queue,
        envs: list[BaseSDKEnv],
        semaphore: asyncio.Semaphore,
    ):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.envs: Dict[str, BaseSDKEnv] = {env.env_name: env for env in envs}
        self.semaphore = semaphore
    
    async def run(self):
        """Main worker loop"""
        while True:
            try:
                task = await self.task_queue.get()
                
                async with self.semaphore:
                    result = await self._execute_task(task)
                
                if result and result.error is None:
                    await self.result_queue.put(result)
                    logger.debug(
                        f"[RESULT] U{result.miner.uid:>3d} │ "
                        f"{result.env:<20} │ "
                        f"{result.score:>6.4f} │ "
                        f"{result.latency_seconds:>6.3f}s"
                    )
                elif result and result.error:
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