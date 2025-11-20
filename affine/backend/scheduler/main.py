"""
Task Scheduler Service - Main Entry Point

Runs the TaskScheduler as an independent background service.
This service generates sampling tasks for all miners periodically.

Usage:
    python -m affine.backend.task_scheduler.main
"""

import os
import asyncio
import signal
from affine.core.setup import setup_logging, logger
from affine.database import init_client, close_client
from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.task_queue import TaskQueueDAO
from affine.backend.scheduler.task_generator import TaskGeneratorService
from affine.affine.backend.scheduler.scheduler import SchedulerService

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()


async def main():
    setup_logging(1)
    logger.info("Starting Task Scheduler Service")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize database
    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1
    
    # Get configuration from environment
    task_generation_interval = int(os.getenv("SCHEDULER_TASK_GENERATION_INTERVAL", "300"))  # 5 minutes
    cleanup_interval = int(os.getenv("SCHEDULER_CLEANUP_INTERVAL", "3600"))  # 1 hour
    max_tasks_per_miner_env = int(os.getenv("SCHEDULER_MAX_TASKS_PER_MINER_ENV", "100"))
    
    # Initialize task generator and scheduler
    scheduler = None
    try:
        # Create DAOs
        sample_results_dao = SampleResultsDAO()
        task_queue_dao = TaskQueueDAO()
        
        # Create TaskGeneratorService
        task_generator = TaskGeneratorService(
            sample_results_dao=sample_results_dao,
            task_queue_dao=task_queue_dao
        )
        
        # Create and start SchedulerService
        scheduler = SchedulerService(
            task_generator=task_generator,
            task_generation_interval=task_generation_interval,
            cleanup_interval=cleanup_interval,
            max_tasks_per_miner_env=max_tasks_per_miner_env
        )
        
        await scheduler.start()
        logger.info(
            f"TaskScheduler started (task_generation_interval={task_generation_interval}s, "
            f"cleanup_interval={cleanup_interval}s)"
        )
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error running TaskScheduler: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
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
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)