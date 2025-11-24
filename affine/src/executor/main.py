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

from affine.src.executor.worker import ExecutorWorker
from affine.core.setup import wallet, logger, setup_logging


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


async def fetch_system_config(api_url: str) -> dict:
    """Fetch system configuration from API.
    
    Returns:
        System config dict with 'environments' key
    """
    import aiohttp
    
    url = f"{api_url}/api/v1/config/environments"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch environments config from API: HTTP {response.status}")
                    return {"environments": ["affine:sat"]}  # Fallback default
                
                config = await response.json()

                if isinstance(config, dict):
                    value = config.get("param_value")
                    if isinstance(value, dict):
                        # Filter environments where enabled_for_scoring=true
                        enabled_envs = [
                            env_name for env_name, env_config in value.items()
                            if isinstance(env_config, dict) and env_config.get("enabled_for_scoring", False)
                        ]
                        
                        if enabled_envs:
                            logger.info(f"Fetched environments from API: {enabled_envs}")
                            return {"environments": enabled_envs}

                logger.exception("Failed to parse environments config")
                
    except Exception as e:
        logger.error(f"Error fetching system config: {e}")
        raise


async def run_service_with_mode(envs: List[str] | None, api_url: str, poll_interval: int, service_mode: bool):
    """Run the executor service.
    
    Args:
        envs: List of environments to execute (None to fetch from API)
        api_url: API base URL
        poll_interval: Polling interval in seconds
        service_mode: If True, run continuously; if False, run once and exit
    """
    
    # Fetch environments from API if not specified
    if not envs:
        logger.info("No environments specified, fetching from API system config...")
        system_config = await fetch_system_config(api_url)
        envs = system_config.get("environments")
    else:
        logger.info(f"Using specified environments: {envs}")
    
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
        
        # Wait for shutdown signal
        if not service_mode:
            # One-time execution (DEFAULT)
            logger.info("Running in one-time mode (default): processing available tasks...")
            await asyncio.sleep(poll_interval * 2)
            manager.print_status()
        else:
            # Continuous service mode (SERVICE_MODE=true)
            logger.info("Running in service mode (continuous). Press Ctrl+C to stop.")
            
            # Periodically print status
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
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
    help="Environments to execute tasks for (e.g., affine:sat). If not specified, fetches from API system config"
)
@click.option(
    "--poll-interval",
    default=None,
    type=int,
    help="Seconds between polling for tasks (default: from EXECUTOR_POLL_INTERVAL or 5)"
)
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(envs, poll_interval, verbosity):
    """
    Affine Executor - Execute sampling tasks for multiple environments.
    
    Each environment runs in its own async worker, polling for tasks and executing them.
    
    Run Mode:
    - Default: One-time execution (processes available tasks once)
    - SERVICE_MODE=true: Continuous service mode (keeps running)
    
    Configuration:
    - EXECUTOR_POLL_INTERVAL: Polling interval in seconds (default: 5)
    - SERVICE_MODE: Run as continuous service (default: false)
    
    If --envs not specified, environments are fetched from API /api/v1/config/environments endpoint.
    """
    # Setup logging if verbosity specified
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    # Get environments from CLI (or None to fetch from API)
    selected_envs = list(envs) if envs else None
    
    # Get API URL from environment variable
    api_base_url = os.getenv("AFFINE_API_URL", "http://localhost:8000")
    
    # Get poll interval (priority: CLI arg > env var > default)
    poll_interval_val = poll_interval if poll_interval is not None else int(os.getenv("EXECUTOR_POLL_INTERVAL", "5"))

    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    logger.info(f"API URL: {api_base_url}")
    logger.info(f"Poll interval: {poll_interval_val}s")
    logger.info(f"Service mode: {service_mode}")
    
    # Run service
    asyncio.run(run_service_with_mode(
        envs=selected_envs,
        api_url=api_base_url,
        poll_interval=poll_interval_val,
        service_mode=service_mode
    ))


if __name__ == "__main__":
    main()