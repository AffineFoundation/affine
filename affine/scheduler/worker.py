import time
import asyncio
import traceback
from typing import Dict, Optional
from affine.tasks import BaseSDKEnv
from affine.models import Result
from affine.scheduler.models import Task
from affine.scheduler.queue import TaskQueue
from affine.scheduler.error_classifier import is_service_error, is_model_error
from affine.setup import logger


class EvaluationWorker:
    """Evaluation worker that executes sampling tasks"""
    
    def __init__(
        self,
        worker_id: str,
        task_queue: TaskQueue,
        result_queue: asyncio.Queue,
        env: BaseSDKEnv,
        samplers: Dict[int, 'MinerSampler'],
        monitor: Optional['SchedulerMonitor'] = None,
    ):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.env = env
        self.samplers = samplers
        self.monitor = monitor
    
    async def run(self):
        """Main worker loop"""
        while True:
            try:
                task = await self.task_queue.get()
                result = await self._execute_task(task)
                
                if result:
                    # If error exists, skip the result
                    if result.error:
                        logger.debug(
                            f"[SKIP]   U{result.miner.uid:>3d} │ "
                            f"{result.env:<20} │ "
                            f"Error (skipped): {result.error}"
                        )
                    # If no error, put result in queue
                    else:
                        await self.result_queue.put(result)
                        logger.debug(
                            f"[RESULT] U{result.miner.uid:>3d} │ "
                            f"{result.env:<20} │ "
                            f"{result.score:>6.4f} │ "
                            f"task_id={result.task_id} │ "
                            f"{result.latency_seconds:>6.3f}s"
                        )
            
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[Worker-{self.worker_id}] Error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task) -> Result:
        """Execute single evaluation task"""
        result = await self.env.evaluate(task.miner, seed=task.seed, task_id=task.task_id)
        return result
    