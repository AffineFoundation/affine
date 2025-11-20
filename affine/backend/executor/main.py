#!/usr/bin/env python3
"""
Executor Main Entry Point

Runs executor workers for multiple environments concurrently.
Each environment gets its own thread/worker.
"""

import signal
import asyncio
import os
import click
from typing import List

from affine.backend.executor.worker import ExecutorWorker
from affine.core.setup import wallet, logger, setup_logging

DEFAULT_ENVS = [
    "affine:sat",
    # "affine:abd",
    # "affine:ded",
]

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
        
        # Create workers first
        self._create_workers()
        
        # Initialize all workers serially (initialization must be serial)
        for worker in self.workers:
            await worker.initialize()
        
        # Start all workers (start() just kicks off background loops, returns immediately)
        for worker in self.workers:
            worker.start()

        self.running = True
        
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
        
        logger.info("ExecutorManager stopped")
    
    async def wait(self):
        """Wait for shutdown signal (workers run in their own loops)."""
        while self.running:
            await asyncio.sleep(1)
    
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


async def run_service(envs: List[str], api_url: str, poll_interval: int, single_run: bool, show_status: bool):
    """Run the executor service."""
    logger.info(f"Starting executor for environments: {envs}")
    
    # Create manager
    manager = ExecutorManager(
        envs=envs,
        api_base_url=api_url,
        poll_interval=poll_interval,
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
        if show_status:
            manager.print_status()
        
        # Wait for shutdown signal
        if single_run:
            logger.info("Single-run mode: waiting for one cycle...")
            await asyncio.sleep(poll_interval * 2)
            manager.print_status()
        else:
            logger.info("Running in continuous mode. Press Ctrl+C to stop.")
            
            # Periodically print status
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    if show_status:
                        manager.print_status()
        
        # Print final status
        manager.print_status()
        
    except Exception as e:
        logger.error(f"Error running executor: {e}", exc_info=True)
        raise
    finally:
        await manager.stop()


@click.command()
@click.option(
    "--envs",
    multiple=True,
    help="Environments to execute tasks for (e.g., affine:sat)"
)
@click.option(
    "--all-envs",
    is_flag=True,
    help="Run all default environments"
)
@click.option(
    "--api-url",
    default=None,
    help="API server URL (default: from AFFINE_API_URL or http://localhost:8000)"
)
@click.option(
    "--poll-interval",
    default=5,
    type=int,
    help="Seconds between polling for tasks"
)
@click.option(
    "-v", "--verbosity",
    default="1",
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
@click.option(
    "--single-run",
    is_flag=True,
    help="Run once and exit (for debugging)"
)
def main(envs, all_envs, api_url, poll_interval, verbosity, single_run):
    """
    Affine Executor - Execute sampling tasks for multiple environments.
    
    Each environment runs in its own async worker, polling for tasks and executing them.
    """
    # Setup logging
    setup_logging(int(verbosity))
    
    # Determine environments
    if all_envs:
        selected_envs = list(DEFAULT_ENVS)
    elif envs:
        selected_envs = list(envs)
    else:
        selected_envs = list(DEFAULT_ENVS)
    
    # Get API URL
    api_base_url = api_url or os.getenv("AFFINE_API_URL", "http://localhost:8000")
    
    # Show status in debug mode
    show_status = int(verbosity) >= 2
    
    # Run service
    asyncio.run(run_service(
        envs=selected_envs,
        api_url=api_base_url,
        poll_interval=poll_interval,
        single_run=single_run,
        show_status=show_status
    ))


if __name__ == "__main__":
    main()