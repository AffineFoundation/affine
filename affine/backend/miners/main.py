"""
Miners Monitor Service - Main Entry Point

Runs the MinersMonitor as an independent background service.
This service monitors miners from the metagraph and updates the database.

Usage:
    python -m affine.backend.miners_monitor.main
"""

import asyncio
import signal
from affine.core.setup import logger, setup_logging
from affine.database import init_client, close_client
from miners_monitor import MinersMonitor

shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()


async def main():
    setup_logging(1)
    logger.info("Starting Miners Monitor Service")

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
    
    # Initialize and start MinersMonitor
    monitor = None
    try:
        monitor = await MinersMonitor.initialize()
        logger.info(f"MinersMonitor started")

        # Wait for shutdown signal
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error running MinersMonitor: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        if monitor:
            try:
                await monitor.stop_background_tasks()
                logger.info("MinersMonitor stopped")
            except Exception as e:
                logger.error(f"Error stopping MinersMonitor: {e}")
        
        try:
            await close_client()
            logger.info("Database client closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    logger.info("Miners Monitor Service shut down successfully")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)