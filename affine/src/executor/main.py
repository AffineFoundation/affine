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


def _format_change(value: int) -> str:
    """Format change value with +/- prefix, or empty string if zero."""
    if value > 0:
        return f"+{value}"
    elif value < 0:
        return str(value)
    return ""


def _format_env_queue(env_name: str, queue_count: int, change: int) -> str:
    """Format environment queue with optional change indicator."""
    env_short = env_name.split(':')[-1]
    change_str = _format_change(change)
    return f"{env_short}={queue_count}({change_str})" if change_str else f"{env_short}={queue_count}"


def _format_env_stats(env_name: str, completed: int, success_rate: int, change: int, running: int, pending: int, fetch_avg_ms: float) -> str:
    """Format environment completion stats with running, pending tasks, and fetch latency."""
    env_short = env_name.split(':')[-1]
    change_str = f" finished:{_format_change(change)}" if change else " finished:0"
    return f"{env_short}@{completed}({success_rate}%{change_str} running:{running} pending:{pending} fetch_avg:{fetch_avg_ms:.0f}ms)"


class ExecutorManager:
    """
    Manages multiple executor workers.
    
    Each environment runs in its own async task/worker.
    """
    
    def __init__(
        self,
        envs: List[str],
        max_concurrent_tasks: int = 5,
    ):
        """
        Initialize ExecutorManager.
        
        Args:
            envs: List of environments to execute tasks for
            max_concurrent_tasks: Maximum concurrent tasks per env (default: 5)
        """
        self.envs = envs
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.workers: List[ExecutorWorker] = []
        self.running = False
        
        # Status tracking for incremental statistics
        self.last_status_time = None
        self.last_queue_stats = {}  # {env: queue_count}
        self.last_completed_stats = {}  # {env: completed_count}
        
        logger.info(
            f"ExecutorManager initialized for {len(envs)} environments "
            f"(max_concurrent: {max_concurrent_tasks})"
        )
    
    def _create_workers(self):
        """Create worker instances for each environment."""
        self.workers = []
        
        for idx, env in enumerate(self.envs):
            worker = ExecutorWorker(
                worker_id=idx,
                env=env,
                wallet=self.wallet,
                max_concurrent_tasks=self.max_concurrent_tasks,
                batch_size=20,  # Fetch 20 tasks per request
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
    
    async def _fetch_queue_stats(self, metrics):
        """Fetch queue statistics from API for all environments."""
        api_client = create_api_client()
        queue_stats = {}
        
        for m in metrics:
            env = m['env']
            try:
                stats_response = await api_client.get(f"/tasks/pool/stats?env={env}")
                queue_stats[env] = stats_response.get('pending_count', 0) if isinstance(stats_response, dict) else 0
            except:
                queue_stats[env] = 0
        
        return queue_stats
    
    async def print_status(self):
        """Print status of all workers in compact format with incremental statistics."""
        import time
        
        try:
            current_time = time.time()
            metrics = self.get_all_metrics()
            
            # Fetch queue statistics
            queue_stats = await self._fetch_queue_stats(metrics)
            total_queue = sum(queue_stats.values())
            time_delta = int(current_time - self.last_status_time) if self.last_status_time else 0
            
            # Calculate queue changes
            env_queue_changes = {
                env: current - self.last_queue_stats.get(env, current)
                for env, current in queue_stats.items()
            }
            total_queue_change = sum(env_queue_changes.values())
            
            # Build environment stats
            env_stats = {}
            for m in metrics:
                env_name = m['env']
                completed, failed = m['tasks_completed'], m['tasks_failed']
                total = completed + failed
                success_rate = int(completed * 100 / total) if total > 0 else 0
                completed_change = completed - self.last_completed_stats.get(env_name, completed)
                
                env_stats[env_name] = {
                    'completed': completed,
                    'success_rate': success_rate,
                    'queue': queue_stats.get(env_name, 0),
                    'queue_change': env_queue_changes.get(env_name, 0),
                    'completed_change': completed_change,
                    'running_tasks': m.get('running_tasks', 0),
                    'pending_tasks': m.get('pending_tasks', 0),
                    'fetch_avg_ms': m.get('avg_fetch_time_ms', 0),
                }
            
            # Format status strings
            queue_details = " ".join(
                _format_env_queue(env, stats['queue'], stats['queue_change'])
                for env, stats in sorted(env_stats.items())
            )
            
            env_stats_str = " ".join(
                _format_env_stats(
                    env,
                    stats['completed'],
                    stats['success_rate'],
                    stats['completed_change'],
                    stats['running_tasks'],
                    stats['pending_tasks'],
                    stats['fetch_avg_ms']
                )
                for env, stats in sorted(env_stats.items())
            )
            
            total_change_str = f"{_format_change(total_queue_change)}" if total_queue_change else ""
            
            logger.info(
                f"[STATUS] total_queue={total_queue}({total_change_str} in {time_delta}s) "
                f"({queue_details}) [{env_stats_str}]"
            )
            
            # Update tracking
            self.last_status_time = current_time
            self.last_queue_stats = queue_stats.copy()
            self.last_completed_stats = {m['env']: m['tasks_completed'] for m in metrics}
            
        except Exception as e:
            logger.error(f"Error printing status: {e}", exc_info=True)
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
                if isinstance(env_config, dict) and env_config.get("enabled_for_sampling", False)
            ]
            
            if enabled_envs:
                logger.info(f"Fetched environments from API: {enabled_envs}")
                return {"environments": enabled_envs}

    raise ValueError("Invalid or empty environments config from API")


async def run_service_with_mode(
    envs: List[str] | None,
    max_concurrent_tasks: int,
    service_mode: bool
):
    """Run the executor service.
    
    Args:
        envs: List of environments to execute (None to fetch from API)
        max_concurrent_tasks: Maximum concurrent tasks per env
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
        max_concurrent_tasks=max_concurrent_tasks,
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
            await asyncio.sleep(10)
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
    "--max-concurrent",
    default=None,
    type=int,
    help="Max concurrent tasks per env (default: from EXECUTOR_MAX_CONCURRENT or 30)"
)
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(envs, max_concurrent, verbosity):
    """
    Affine Executor - Execute sampling tasks for multiple environments.
    
    Each environment runs with concurrent task execution using a queue-driven model.
    Tasks are fetched continuously to maintain a buffer in the queue.
    
    Run Mode:
    - Default: One-time execution (processes available tasks once)
    - SERVICE_MODE=true: Continuous service mode (keeps running)
    
    Configuration:
    - EXECUTOR_MAX_CONCURRENT: Max concurrent tasks per env (default: 30)
    - SERVICE_MODE: Run as continuous service (default: false)
    
    If --envs not specified, environments are fetched from API /api/v1/config/environments endpoint.
    """
    # Setup logging if verbosity specified
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    # Get environments from CLI (or None to fetch from API)
    selected_envs = list(envs) if envs else None
    
    # Get max concurrent tasks (priority: CLI arg > env var > default)
    max_concurrent_val = max_concurrent if max_concurrent is not None else int(os.getenv("EXECUTOR_MAX_CONCURRENT", "30"))

    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    logger.info(f"Max concurrent tasks: {max_concurrent_val} per env")
    logger.info(f"Service mode: {service_mode}")
    
    # Run service
    asyncio.run(run_service_with_mode(
        envs=selected_envs,
        max_concurrent_tasks=max_concurrent_val,
        service_mode=service_mode
    ))


if __name__ == "__main__":
    main()
