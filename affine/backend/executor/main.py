#!/usr/bin/env python3
"""
Executor Main Entry Point

Runs executor workers for multiple environments concurrently.
Each environment gets its own thread/worker.

Usage:
    python -m affine.backend.executor.main --envs affine:sat affine:abd
    python -m affine.backend.executor.main --all-envs
    python -m affine.backend.executor.main --debug --envs affine:sat
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from typing import List, Optional

from affine.backend.executor.worker import ExecutorWorker
from affine.core.setup import wallet

# Default environments to run if not specified
DEFAULT_ENVS = [
    "affine:sat",
    "affine:abd",
    "affine:ded",
]

logger = logging.getLogger(__name__)


class ExecutorManager:
    """
    Manages multiple executor workers.
    
    Each environment runs in its own async task/worker.
    """
    
    def __init__(
        self,
        envs: List[str],
        api_base_url: str = "http://localhost:8000",
        poll_interval: int = 5,
    ):
        """
        Initialize ExecutorManager.
        
        Args:
            envs: List of environments to execute tasks for
            api_base_url: API server URL
            poll_interval: How often workers poll for tasks
        """
        self.envs = envs
        self.api_base_url = api_base_url
        self.poll_interval = poll_interval
        
        self.workers: List[ExecutorWorker] = []
        self.worker_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info(f"ExecutorManager initialized for {len(envs)} environments")
    
    def _create_workers(self):
        """Create worker instances for each environment."""
        self.workers = []
        
        for idx, env in enumerate(self.envs):
            worker = ExecutorWorker(
                worker_id=idx,
                env=env,
                api_base_url=self.api_base_url,
                poll_interval=self.poll_interval,
            )
            self.workers.append(worker)
            logger.info(f"Created worker {idx} for {env}")
    
    async def start(self):
        """Start all workers."""
        if self.running:
            logger.warning("ExecutorManager already running")
            return
        
        logger.info("Starting ExecutorManager...")
        
        # Validate wallet
        if not wallet:
            logger.error("No wallet configured. Set WALLET_NAME and WALLET_HOTKEY environment variables.")
            raise RuntimeError("Wallet not configured")
        
        logger.info(f"Using wallet hotkey: {wallet.hotkey.ss58_address[:16]}...")
        
        self.running = True
        self._create_workers()
        
        # Start all workers as async tasks
        self.worker_tasks = []
        for worker in self.workers:
            task = asyncio.create_task(worker.start())
            self.worker_tasks.append(task)
        
        logger.info(f"Started {len(self.workers)} workers")
    
    async def stop(self):
        """Stop all workers gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping ExecutorManager...")
        self.running = False
        
        # Stop all workers
        for worker in self.workers:
            await worker.stop()
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("ExecutorManager stopped")
    
    async def wait(self):
        """Wait for all workers to complete."""
        if self.worker_tasks:
            try:
                await asyncio.gather(*self.worker_tasks)
            except asyncio.CancelledError:
                pass
    
    def get_all_metrics(self):
        """Get metrics from all workers."""
        return [worker.get_metrics() for worker in self.workers]
    
    def print_status(self):
        """Print status of all workers."""
        metrics = self.get_all_metrics()
        
        print("\n" + "=" * 60)
        print("Executor Status")
        print("=" * 60)
        
        for m in metrics:
            print(f"Worker {m['worker_id']} ({m['env']}):")
            print(f"  Running: {m['running']}")
            print(f"  Tasks Completed: {m['tasks_completed']}")
            print(f"  Tasks Failed: {m['tasks_failed']}")
            print(f"  Avg Execution Time: {m['avg_execution_time']:.2f}s")
            if m['last_task_at']:
                import time
                elapsed = time.time() - m['last_task_at']
                print(f"  Last Task: {elapsed:.1f}s ago")
            print()
        
        print("=" * 60)


async def main(args):
    """Main entry point."""
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    
    # Determine which environments to run
    if args.all_envs:
        envs = DEFAULT_ENVS
    else:
        envs = args.envs or DEFAULT_ENVS
    
    logger.info(f"Starting executor for environments: {envs}")
    
    # Get API URL from args or environment
    api_url = args.api_url or os.getenv("AFFINE_API_URL", "http://localhost:8000")
    
    # Create manager
    manager = ExecutorManager(
        envs=envs,
        api_base_url=api_url,
        poll_interval=args.poll_interval,
    )
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
    
    try:
        # Start manager
        await manager.start()
        
        # Print initial status
        if args.debug:
            manager.print_status()
        
        # Wait for shutdown signal
        if args.single_run:
            # For debugging: run once and exit
            logger.info("Single-run mode: waiting for one cycle...")
            await asyncio.sleep(args.poll_interval * 2)
            manager.print_status()
        else:
            # Normal mode: run until shutdown
            logger.info("Running in continuous mode. Press Ctrl+C to stop.")
            
            # Periodically print status
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    if args.debug:
                        manager.print_status()
        
        # Print final status
        manager.print_status()
        
    except Exception as e:
        logger.error(f"Error running executor: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Affine Executor - Run sampling tasks for multiple environments"
    )
    
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environments to run (e.g., affine:sat affine:abd)"
    )
    
    parser.add_argument(
        "--all-envs",
        action="store_true",
        help="Run all default environments"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API server URL (default: http://localhost:8000 or AFFINE_API_URL env)"
    )
    
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between polling for tasks (default: 5)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run once and exit (for debugging)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))