"""
Scorer Service - Main Entry Point

Runs the Scorer as an independent service or one-time execution.
Calculates miner weights using the four-stage scoring algorithm.
"""

import os
import asyncio
import click
import time
import aiohttp

from affine.core.setup import setup_logging, logger
from affine.database import init_client, close_client
from affine.database.dao.miner_scores import MinerScoresDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from scorer import Scorer
from config import ScorerConfig


async def fetch_scoring_data(api_base_url: str) -> dict:
    """Fetch scoring data from API."""
    url = f"{api_base_url}/api/v1/samples/scoring"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch scoring data: HTTP {response.status}")
            
            data = await response.json()
            logger.info(f"Fetched scoring data for {len(data)} miners")
            return data


async def fetch_system_config(api_base_url: str) -> dict:
    """Fetch system configuration from API."""
    url = f"{api_base_url}/api/v1/config/scoring"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.warning("Failed to fetch system config, using defaults")
                return {"environments": ["affine:sat", "affine:abd", "affine:ded"]}
            
            return await response.json()


async def run_scoring_once(
    api_base_url: str,
    save_to_db: bool
):
    """Run scoring calculation once."""
    start_time = time.time()
    
    # Use default config (constants)
    config = ScorerConfig()
    scorer = Scorer(config)
    
    # Fetch data
    logger.info("Fetching data from API...")
    scoring_data = await fetch_scoring_data(api_base_url)
    system_config = await fetch_system_config(api_base_url)
    
    # Extract environments
    environments = system_config.get("environments")
    
    # Use current timestamp as block number
    block_number = int(time.time())
    
    # Calculate scores
    logger.info("Starting scoring calculation...")
    result = scorer.calculate_scores(
        scoring_data=scoring_data,
        environments=environments,
        block_number=block_number,
        print_summary=True
    )
    
    # Save to database if requested
    if save_to_db:
        logger.info("Saving results to database...")
        miner_scores_dao = MinerScoresDAO()
        score_snapshots_dao = ScoreSnapshotsDAO()
        
        await scorer.save_results(
            result=result,
            miner_scores_dao=miner_scores_dao,
            score_snapshots_dao=score_snapshots_dao
        )
        logger.info("Results saved successfully")
    
    elapsed = time.time() - start_time
    logger.info(f"Scoring completed in {elapsed:.2f}s")
    
    # Print summary
    summary = result.get_summary()
    logger.info(f"Summary: {summary}")
    
    return result


async def run_service_with_mode(api_base_url: str, save_to_db: bool, service_mode: bool):
    """Run the scorer service.
    
    Args:
        api_base_url: API base URL
        save_to_db: Whether to save results to database
        service_mode: If True, run continuously; if False, run once and exit
    """
    logger.info("Starting Scorer Service")
    
    # Initialize database if saving results
    if save_to_db:
        try:
            await init_client()
            logger.info("Database client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    try:
        if not service_mode:
            # Run once and exit (DEFAULT)
            logger.info("Running in one-time mode (default)")
            await run_scoring_once(api_base_url, save_to_db)
        else:
            # Run continuously every 30 minutes (SERVICE_MODE=true)
            logger.info("Running in service mode (continuous, every 30 minutes)")
            while True:
                try:
                    await run_scoring_once(api_base_url, save_to_db)
                    logger.info("Waiting 30 minutes until next run...")
                    await asyncio.sleep(30 * 60)  # 30 minutes
                except Exception as e:
                    logger.error(f"Error in scoring cycle: {e}", exc_info=True)
                    logger.info("Waiting 30 minutes before retry...")
                    await asyncio.sleep(30 * 60)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error running Scorer: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if save_to_db:
            try:
                await close_client()
                logger.info("Database client closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
    
    logger.info("Scorer Service completed successfully")


@click.command()
@click.option(
    "-v", "--verbosity",
    default="1",
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(verbosity):
    """
    Affine Scorer - Calculate miner weights using four-stage algorithm.
    
    This service fetches scoring data from the API and calculates normalized
    weights for miners using a four-stage algorithm with Pareto filtering.
    
    Run Mode:
    - Default: One-time execution (calculates scores once and exits)
    - SERVICE_MODE=true: Continuous service mode (runs every 30 minutes)
    
    Configuration:
    - AFFINE_API_URL: API base URL (default: http://localhost:8000)
    - SCORER_SAVE_TO_DB: Enable database saving (default: false)
    - SERVICE_MODE: Run as continuous service (default: false)
    - All scoring parameters are constants in config.py
    """
    # Setup logging
    setup_logging(int(verbosity))
    
    # Get API URL from environment variable
    api_base_url = os.getenv("AFFINE_API_URL", "http://localhost:8000")
    
    # Check if should save to database
    save_to_db = os.getenv("SCORER_SAVE_TO_DB", "false").lower() in ("true", "1", "yes")
    
    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    logger.info(f"API URL: {api_base_url}")
    if save_to_db:
        logger.info("Database saving enabled (SCORER_SAVE_TO_DB=true)")
    logger.info(f"Service mode: {service_mode}")
    
    # Run service
    asyncio.run(run_service_with_mode(
        api_base_url=api_base_url,
        save_to_db=save_to_db,
        service_mode=service_mode
    ))


if __name__ == "__main__":
    main()