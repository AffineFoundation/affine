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
                    # Classify error type using centralized error classifier
                    service_error = is_service_error(result.error)
                    model_error = is_model_error(result.error)
                    
                    # Handle service errors - pause sampling and skip result
                    if service_error:
                        # Notify sampler to handle error (for pause logic)
                        sampler = self.samplers.get(task.uid)
                        if sampler:
                            sampler.handle_error(result.error)
                        
                        # Record error to monitor (for statistics)
                        if self.monitor:
                            self.monitor.record_error(task.uid, result.error)
                        
                        logger.debug(
                            f"[SKIP]   U{result.miner.uid:>3d} │ "
                            f"{result.env:<20} │ "
                            f"Service error (skipped): {result.error}"
                        )
                    
                    # Handle model errors or successful results - keep result
                    else:
                        await self.result_queue.put(result)
                        
                        if model_error:
                            # Record model error to monitor (for statistics)
                            if self.monitor:
                                self.monitor.record_error(task.uid, result.error)
                            
                            logger.debug(
                                f"[MODEL]  U{result.miner.uid:>3d} │ "
                                f"{result.env:<20} │ "
                                f"{result.score:>6.4f} │ "
                                f"Model error (recorded): {result.error[:50]}"
                            )
                        else:
                            # Successful sample - reset error state
                            sampler = self.samplers.get(task.uid)
                            if sampler:
                                sampler.reset_error_state()
                            
                            logger.debug(
                                f"[RESULT] U{result.miner.uid:>3d} │ "
                                f"{result.env:<20} │ "
                                f"{result.score:>6.4f} │ "
                                f"{result.latency_seconds:>6.3f}s"
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
            result = await self.env.evaluate(task.miner, seed=task.seed)
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
    