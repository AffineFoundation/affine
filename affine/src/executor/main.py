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
import bittensor as bt

from affine.src.executor.worker import ExecutorWorker
from affine.core.setup import logger, setup_logging
from affine.utils.api_client import create_api_client


class ExecutorManager:
    """
    Manages multiple executor workers.
    
    Each environment runs in its own async task/worker.
    """
    
    def __init__(
        self,
        envs: List[str],
        poll_interval: int = 5,
    ):
        """
        Initialize ExecutorManager.
        
        Args:
            envs: List of environments to execute tasks for
            poll_interval: How often workers poll for tasks
        """
        self.envs = envs
        self.poll_interval = poll_interval
        
        self.workers: List[ExecutorWorker] = []
        self.running = False
        
        # Status tracking for incremental statistics
        self.last_status_time = None
        self.last_queue_stats = {}  # {env: queue_count}
        self.last_completed_stats = {}  # {env: completed_count}
        
        logger.info(f"ExecutorManager initialized for {len(envs)} environments")
    
    def _create_workers(self):
        """Create worker instances for each environment."""
        self.workers = []
        
        for idx, env in enumerate(self.envs):
            worker = ExecutorWorker(
                worker_id=idx,
                env=env,
                wallet=self.wallet,
                poll_interval=self.poll_interval,
            )
            self.workers.append(worker)
            logger.debug(f"Created worker {idx} for {env}")
    
    async def start(self):
        """Start all workers."""
        if self.running:
            logger.warning("ExecutorManager already running")
            return
        
        logger.info("Starting ExecutorManager...")

        coldkey = os.getenv("BT_WALLET_COLD", "default")
        hotkey = os.getenv("BT_WALLET_HOT", "default")
        self.wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        # Validate wallet
        if not self.wallet:
            logger.error("No wallet configured. Set WALLET_NAME and WALLET_HOTKEY environment variables.")
            raise RuntimeError("Wallet not configured")
        
        logger.info(f"Using wallet hotkey: {self.wallet.hotkey.ss58_address[:16]}...")
        
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
    
    async def print_status(self):
        """Print status of all workers in compact format with incremental statistics."""
        import time
        from affine.utils.api_client import create_api_client
        
        try:
            current_time = time.time()
            
            # Get worker metrics
            metrics = self.get_all_metrics()
            
            # Fetch statistics from API
            api_client = create_api_client()
            
            # Get task queue statistics from API
            try:
                queue_stats = {}
                
                for m in metrics:
                    env = m['env']
                    try:
                        # Use API endpoint to get queue stats
                        stats_response = await api_client.get(f"/tasks/pool/stats?env={env}")
                        if isinstance(stats_response, dict):
                            queue_stats[env] = stats_response.get('pending_count', 0)
                        else:
                            queue_stats[env] = 0
                    except:
                        queue_stats[env] = 0
                
                total_queue = sum(queue_stats.values())
            except:
                total_queue = 0
                queue_stats = {}
            
            # Calculate incremental changes
            time_delta = int(current_time - self.last_status_time) if self.last_status_time else 0
            
            # Queue changes (↓ dequeued, ↑ enqueued)
            total_dequeued = 0
            total_enqueued = 0
            env_queue_changes = {}
            
            for env, current_queue in queue_stats.items():
                last_queue = self.last_queue_stats.get(env, current_queue)
                change = current_queue - last_queue
                
                if change < 0:
                    # Queue decreased (dequeued)
                    dequeued = abs(change)
                    enqueued = 0
                else:
                    # Queue increased (enqueued)
                    dequeued = 0
                    enqueued = change
                
                env_queue_changes[env] = (dequeued, enqueued)
                total_dequeued += dequeued
                total_enqueued += enqueued
            
            # Build environment stats from worker metrics
            env_stats = {}
            total_completed = 0
            total_failed = 0
            
            for m in metrics:
                env_name = m['env']
                completed = m['tasks_completed']
                failed = m['tasks_failed']
                total = completed + failed
                success_rate = int(completed * 100 / total) if total > 0 else 0
                
                env_stats[env_name] = {
                    'completed': completed,
                    'failed': failed,
                    'success_rate': success_rate,
                    'queue': queue_stats.get(env_name, 0)
                }
                
                total_completed += completed
                total_failed += failed
            
            # Format queue details with incremental changes
            # Format: env=current_queue(↓dequeued ↑enqueued)
            queue_details_parts = []
            for env, stats in sorted(env_stats.items()):
                env_short = env.split(':')[-1]
                current_q = stats['queue']
                deq, enq = env_queue_changes.get(env, (0, 0))
                queue_details_parts.append(f"{env_short}={current_q}(↓{deq} ↑{enq})")
            
            queue_details = " ".join(queue_details_parts)
            
            # Format environment completion stats: env@completed(success_rate%)
            env_stats_str = " ".join([
                f"{env.split(':')[-1]}@{stats['completed']}({stats['success_rate']}%)"
                for env, stats in sorted(env_stats.items())
            ])
            
            # Build compact status line
            # [STATUS] total_queue=N(↓X ↑Y in Zs) (env_queue_details) result_queue=0 enqueued=total dequeued=total [env@completed(rate%)]
            logger.info(
                f"[STATUS] total_queue={total_queue}(↓{total_dequeued} ↑{total_enqueued} in {time_delta}s) "
                f"({queue_details}) "
                f"result_queue=0 enqueued={total_enqueued} dequeued={total_dequeued} "
                f"[{env_stats_str}]"
            )
            
            # Update tracking for next iteration
            self.last_status_time = current_time
            self.last_queue_stats = queue_stats.copy()
            self.last_completed_stats = {m['env']: m['tasks_completed'] for m in metrics}
            
        except Exception as e:
            logger.error(f"Error printing status: {e}", exc_info=True)
            # Fallback to simple metrics display
            metrics = self.get_all_metrics()
            logger.info(f"[STATUS] workers={len(metrics)} completed={sum(m['tasks_completed'] for m in metrics)}")


async def fetch_system_config() -> dict:
    """Fetch system configuration from API.
    
    Returns:
        System config dict with 'environments' key
        
    Raises:
        Exception: If failed to fetch config from API
    """
    api_client = create_api_client()
    config = await api_client.get("/config/environments")

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

    raise ValueError("Invalid or empty environments config from API")


async def run_service_with_mode(envs: List[str] | None, poll_interval: int, service_mode: bool):
    """Run the executor service.
    
    Args:
        envs: List of environments to execute (None to fetch from API)
        poll_interval: Polling interval in seconds
        service_mode: If True, run continuously; if False, run once and exit
    """
    
    # Fetch environments from API if not specified
    if not envs:
        logger.info("No environments specified, fetching from API system config...")
        system_config = await fetch_system_config()
        envs = system_config.get("environments")
    else:
        logger.info(f"Using specified environments: {envs}")
    
    # Create manager
    manager = ExecutorManager(
        envs=envs,
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
            await manager.print_status()
        else:
            # Continuous service mode (SERVICE_MODE=true)
            logger.info("Running in service mode (continuous). Press Ctrl+C to stop.")
            
            # Periodically print status
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    await manager.print_status()
        
        # Print final status
        await manager.print_status()
        
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
    
    # Get poll interval (priority: CLI arg > env var > default)
    poll_interval_val = poll_interval if poll_interval is not None else int(os.getenv("EXECUTOR_POLL_INTERVAL", "5"))

    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    logger.info(f"Poll interval: {poll_interval_val}s")
    logger.info(f"Service mode: {service_mode}")
    
    # Run service
    asyncio.run(run_service_with_mode(
        envs=selected_envs,
        poll_interval=poll_interval_val,
        service_mode=service_mode
    ))


if __name__ == "__main__":
    main()