"""
Worker Process Management - Subprocess wrapper and entry point
"""

import asyncio
import time
import queue
import multiprocessing
from typing import Optional
import bittensor as bt
from affine.core.setup import logger


async def stats_reporter(worker, stats_queue: multiprocessing.Queue, env: str, interval: int = 5):
    """Report worker stats to main process periodically.
    
    Args:
        worker: ExecutorWorker instance
        stats_queue: Queue for sending stats to main process
        env: Environment name
        interval: Reporting interval in seconds
    """
    while worker.running:
        try:
            metrics = worker.get_metrics()
            metrics['reported_at'] = time.time()
            
            try:
                stats_queue.put_nowait(metrics)
            except queue.Full:
                pass
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"[{env}] Stats reporter error: {e}")
            await asyncio.sleep(1)


def run_worker_subprocess(
    worker_id: int,
    env: str,
    wallet_name: str,
    wallet_hotkey: str,
    max_concurrent_tasks: int,
    batch_size: int,
    stats_queue: multiprocessing.Queue,
    verbosity: int = 1,
):
    """Run worker in subprocess.
    
    This is the entry point for each subprocess. It creates its own event loop
    and runs the worker independently.
    
    Args:
        worker_id: Worker identifier
        env: Environment name
        wallet_name: Wallet cold key name
        wallet_hotkey: Wallet hot key name
        max_concurrent_tasks: Max concurrent tasks
        batch_size: Fetch batch size
        stats_queue: Queue for stats reporting
        verbosity: Logging verbosity level
    """
    from affine.src.executor.worker import ExecutorWorker
    from affine.core.setup import setup_logging
    
    # Setup logging in subprocess with explicit component name
    setup_logging(verbosity, component="executor")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    worker = None
    
    try:
        wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
        
        worker = ExecutorWorker(
            worker_id=worker_id,
            env=env,
            wallet=wallet,
            max_concurrent_tasks=max_concurrent_tasks,
            batch_size=batch_size,
        )
        
        loop.run_until_complete(worker.initialize())
        
        # Create tasks AFTER starting the event loop
        async def start_worker():
            """Start worker and create all async tasks"""
            worker.start()
            asyncio.create_task(stats_reporter(worker, stats_queue, env))
        
        # Schedule worker startup
        loop.create_task(start_worker())
        
        loop.run_forever()
        
    except KeyboardInterrupt:
        logger.info(f"Worker process {env} received interrupt signal")
    except Exception as e:
        logger.error(f"Worker process {env} failed: {e}", exc_info=True)
    finally:
        # Cleanup sequence: stop worker -> cleanup env -> cancel tasks -> close loop
        
        # 1. Stop worker if it was started (running = True)
        if worker and worker.running:
            try:
                loop.run_until_complete(worker.stop())
            except Exception as e:
                logger.error(f"Error stopping worker {env}: {e}")
        
        # 3. Cancel all remaining pending tasks
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            
            # Wait for all tasks to complete cancellation
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            logger.warning(f"Error cancelling tasks for {env}: {e}")
        
        # 4. Close the event loop
        try:
            if not loop.is_closed():
                loop.close()
        except Exception as e:
            logger.warning(f"Error closing event loop for {env}: {e}")


class WorkerProcess:
    """Wrapper for worker subprocess."""
    
    def __init__(
        self,
        worker_id: int,
        env: str,
        wallet_name: str,
        wallet_hotkey: str,
        max_concurrent_tasks: int,
        batch_size: int,
        stats_queue: multiprocessing.Queue,
        verbosity: int = 1,
    ):
        self.worker_id = worker_id
        self.env = env
        self.wallet_name = wallet_name
        self.wallet_hotkey = wallet_hotkey
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        self.stats_queue = stats_queue
        self.verbosity = verbosity
        
        self.process: Optional[multiprocessing.Process] = None
        self.last_heartbeat: float = 0
    
    def start(self):
        """Start the worker subprocess."""
        self.process = multiprocessing.Process(
            target=run_worker_subprocess,
            args=(
                self.worker_id,
                self.env,
                self.wallet_name,
                self.wallet_hotkey,
                self.max_concurrent_tasks,
                self.batch_size,
                self.stats_queue,
                self.verbosity,
            ),
            name=f"Worker-{self.env}",
        )
        self.process.start()
        logger.info(f"Started worker process for {self.env} (PID: {self.process.pid})")
    
    def is_alive(self) -> bool:
        """Check if process is alive."""
        return self.process is not None and self.process.is_alive()
    
    def terminate(self):
        """Terminate the worker process."""
        if self.process:
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                logger.warning(f"Force killing worker {self.env}")
                self.process.kill()