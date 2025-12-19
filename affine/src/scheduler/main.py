"""
Task Scheduler Service - Main Entry Point

Runs the TaskScheduler as an independent background service.
This service generates sampling tasks for all miners periodically.
"""

import os
import asyncio
import signal
import click
from affine.core.setup import setup_logging, logger
from affine.database import init_client, close_client
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.task_pool import TaskPoolDAO
from .task_generator import TaskGeneratorService
from .scheduler import SchedulerService
from .sampling_scheduler import SamplingScheduler


async def run_service(task_interval: int, cleanup_interval: int, max_tasks: int):
    """Run the task scheduler service."""
    logger.info("Starting Task Scheduler Service")
    
    # Initialize database
    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Setup signal handlers
    shutdown_event = asyncio.Event()
    
    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
    
    # Initialize task generator and schedulers
    scheduler = None
    sampling_scheduler = None
    try:
        # Create DAOs
        sample_results_dao = SampleResultsDAO()
        task_pool_dao = TaskPoolDAO()
        
        # Create TaskGeneratorService
        task_generator = TaskGeneratorService(
            sample_results_dao=sample_results_dao,
            task_pool_dao=task_pool_dao
        )
        
        # Create and start SchedulerService
        scheduler = SchedulerService(
            task_generator=task_generator,
            task_generation_interval=task_interval,
            cleanup_interval=cleanup_interval,
            max_tasks_per_miner_env=max_tasks
        )
        
        await scheduler.start()
        logger.info(
            f"TaskScheduler started (task_interval={task_interval}s, "
            f"cleanup_interval={cleanup_interval}s, max_tasks={max_tasks})"
        )
        
        # Create and start SamplingScheduler
        sampling_scheduler = SamplingScheduler()
        await sampling_scheduler.start()
        logger.info("SamplingScheduler started for dynamic task rotation")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error running TaskScheduler: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if sampling_scheduler:
            try:
                await sampling_scheduler.stop()
                logger.info("SamplingScheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping SamplingScheduler: {e}")
        
        if scheduler:
            try:
                await scheduler.stop()
                logger.info("TaskScheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping TaskScheduler: {e}")
        
        try:
            await close_client()
            logger.info("Database client closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    logger.info("Task Scheduler Service shut down successfully")


@click.command()
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(verbosity):
    """
    Affine Task Scheduler - Generate sampling tasks for miners.
    
    This service periodically generates sampling tasks for all active miners
    and performs cleanup of old tasks.
    """
    # Setup logging if verbosity specified
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    # Override with environment variables if present
    task_interval = int(os.getenv("SCHEDULER_TASK_GENERATION_INTERVAL", "600"))
    cleanup_interval = int(os.getenv("SCHEDULER_CLEANUP_INTERVAL", "300"))
    max_tasks = int(os.getenv("SCHEDULER_MAX_TASKS_PER_MINER_ENV", "300"))

    # Run service
    asyncio.run(run_service(
        task_interval=task_interval,
        cleanup_interval=cleanup_interval,
        max_tasks=max_tasks
    ))


if __name__ == "__main__":
    main()