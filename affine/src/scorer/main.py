"""
Scorer Service - Main Entry Point

Runs the Scorer as an independent service or one-time execution.
Calculates miner weights using the four-stage scoring algorithm.
"""

import os
import asyncio
import click
import time

from affine.core.setup import setup_logging, logger
from affine.database import init_client, close_client
from affine.database.dao.miner_scores import MinerScoresDAO
from affine.database.dao.score_snapshots import ScoreSnapshotsDAO
from affine.database.dao.scores import ScoresDAO
from affine.src.scorer.scorer import Scorer
from affine.src.scorer.config import ScorerConfig
from affine.utils.subtensor import get_subtensor
from affine.utils.api_client import create_api_client


async def fetch_scoring_data() -> dict:
    """Fetch scoring data from API with default timeout."""
    api_client = create_api_client()
    
    logger.info("Fetching scoring data from API...")
    data = await api_client.get("/samples/scoring")
    
    # Check for API error response
    if isinstance(data, dict) and "success" in data and data.get("success") is False:
        error_msg = data.get("error", "Unknown API error")
        status_code = data.get("status_code", "unknown")
        logger.error(f"API returned error response: {error_msg} (status: {status_code})")
        raise RuntimeError(f"Failed to fetch scoring data: {error_msg}")
    
    return data


async def fetch_system_config() -> dict:
    """Fetch system configuration from API.
    
    Returns:
        System config dict with 'environments' key
    """
    api_client = create_api_client()
    
    try:
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

        logger.exception("Failed to parse environments config")
                
    except Exception as e:
        logger.error(f"Error fetching system config: {e}")
        raise



async def run_scoring_once(save_to_db: bool):
    """Run scoring calculation once."""
    start_time = time.time()
    
    # Use default config (constants)
    config = ScorerConfig()
    scorer = Scorer(config)
    
    # Fetch data
    logger.info("Fetching data from API...")
    scoring_data = await fetch_scoring_data()
    system_config = await fetch_system_config()
    
    # Extract environments
    environments = system_config.get("environments")
    logger.info(f"environments: {environments}")
    
    # Get current block number from Bittensor
    logger.info("Fetching current block number from Bittensor...")
    subtensor = await get_subtensor()
    block_number = await subtensor.get_current_block()
    logger.info(f"Current block number: {block_number}")
    
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
        scores_dao = ScoresDAO()
        
        await scorer.save_results(
            result=result,
            miner_scores_dao=miner_scores_dao,
            score_snapshots_dao=score_snapshots_dao,
            scores_dao=scores_dao
        )
        logger.info("Results saved successfully")
    
    elapsed = time.time() - start_time
    logger.info(f"Scoring completed in {elapsed:.2f}s")
    
    # Print summary
    summary = result.get_summary()
    logger.info(f"Summary: {summary}")
    
    return result


async def run_service_with_mode(save_to_db: bool, service_mode: bool):
    """Run the scorer service.
    
    Args:
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
            await run_scoring_once(save_to_db)
        else:
            # Run continuously every 30 minutes (SERVICE_MODE=true)
            logger.info("Running in service mode (continuous, every 30 minutes)")
            while True:
                try:
                    await run_scoring_once(save_to_db)
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
def main():
    """
    Affine Scorer - Calculate miner weights using four-stage algorithm.
    
    This service fetches scoring data from the API and calculates normalized
    weights for miners using a four-stage algorithm with Pareto filtering.
    
    Run Mode:
    - Default: One-time execution (calculates scores once and exits)
    - SERVICE_MODE=true: Continuous service mode (runs every 30 minutes)
    
    Configuration:
    - SCORER_SAVE_TO_DB: Enable database saving (default: false)
    - SERVICE_MODE: Run as continuous service (default: false)
    - All scoring parameters are constants in config.py
    """
    # Check if should save to database
    save_to_db = os.getenv("SCORER_SAVE_TO_DB", "false").lower() in ("true", "1", "yes")
    
    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    if save_to_db:
        logger.info("Database saving enabled (SCORER_SAVE_TO_DB=true)")
    logger.info(f"Service mode: {service_mode}")
    
    # Run service
    asyncio.run(run_service_with_mode(
        save_to_db=save_to_db,
        service_mode=service_mode
    ))


if __name__ == "__main__":
    main()