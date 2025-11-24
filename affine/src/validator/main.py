#!/usr/bin/env python3
"""
Validator Service Main Entry Point

Fetches scores from backend API and sets weights on blockchain.
"""

import asyncio
import os
import time
import signal
import click
from typing import Dict, List, Optional, Tuple

from affine.src.validator.weight_setter import WeightSetter
from affine.core.setup import logger, setup_logging
from affine.utils.api_client import create_api_client


class ValidatorService:
    """
    Validator Service - Fetches scores and sets weights.
    
    Main loop:
    1. Fetch latest scores from API
    2. Get active miners from metagraph
    3. Convert hotkey scores to UID weights
    4. Set weights on blockchain
    """
    
    def __init__(
        self,
        wallet=None,
        netuid: int = 120,
    ):
        """
        Initialize ValidatorService.
        
        Args:
            wallet: Bittensor wallet (if None, loads from environment)
            netuid: Subnet UID
        """
        # API client
        self.api_client = create_api_client()
        
        # Wallet and blockchain
        self.wallet = wallet
        self.netuid = netuid
        
        # Weight setter
        self.weight_setter = WeightSetter(
            wallet=wallet,
            netuid=netuid,
            max_retries=3,
            retry_delay=10.0,
            min_interval=300.0,  # 5 minutes minimum between sets
        )
        
        # State
        self.running = False
        self.last_run_at: Optional[float] = None
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        
        logger.info(f"ValidatorService initialized")
    
    async def fetch_latest_scores(self) -> Optional[Dict]:
        """
        Fetch latest scores from backend API.
        
        Returns:
            Dictionary with scores data or None if failed
        """
        try:
            response = await self.api_client.get("/scores/latest")
            
            # Check for API error response
            if isinstance(response, dict) and "success" in response and response.get("success") is False:
                logger.warning("No scores available from backend")
                return None
            
            scores = response.get("scores", [])
            if not scores:
                logger.warning("Empty scores from backend")
                return None
            
            logger.info(
                f"Fetched {len(scores)} miner scores "
                f"(block: {response.get('block_number', 'unknown')})"
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error fetching scores: {e}")
            return None
    
    async def get_metagraph_hotkey_to_uid(self) -> Dict[str, int]:
        """
        Get hotkey to UID mapping from metagraph.
        
        Returns:
            Dictionary mapping hotkey -> uid
        """
        try:
            import bittensor as bt
            
            subtensor = bt.subtensor()
            metagraph = subtensor.metagraph(self.netuid)
            
            hotkey_to_uid = {}
            for uid in range(len(metagraph.hotkeys)):
                hotkey = metagraph.hotkeys[uid]
                hotkey_to_uid[hotkey] = uid
            
            logger.info(f"Loaded {len(hotkey_to_uid)} hotkeys from metagraph")
            return hotkey_to_uid
        
        except ImportError:
            logger.error("bittensor not installed, cannot get metagraph")
            return {}
        except Exception as e:
            logger.error(f"Error getting metagraph: {e}")
            return {}
    
    async def get_active_miners_from_api(self) -> Dict[str, int]:
        """
        Get active miners from API (fallback if metagraph unavailable).
        
        Returns:
            Dictionary mapping hotkey -> uid
        """
        try:
            response = await self.api_client.get("/miners/active")
            
            # Check for API error response
            if isinstance(response, dict) and "success" in response and response.get("success") is False:
                return {}
            
            if response and isinstance(response, list):
                hotkey_to_uid = {
                    miner["hotkey"]: miner["uid"]
                    for miner in response
                    if "hotkey" in miner and "uid" in miner
                }
                logger.info(f"Loaded {len(hotkey_to_uid)} miners from API")
                return hotkey_to_uid
            
            return {}
        
        except Exception as e:
            logger.error(f"Error fetching miners from API: {e}")
            return {}
    
    def scores_to_uid_weights(
        self,
        scores: List[Dict],
        hotkey_to_uid: Dict[str, int],
    ) -> Tuple[List[int], List[float]]:
        """
        Convert hotkey-based scores to UID-based weights.
        
        IMPORTANT: Returns weights for ALL UIDs in metagraph, with 0 for miners
        without scores. This ensures that miners not in the top scores explicitly
        get their weights reset to 0 on chain.
        
        Args:
            scores: List of score dictionaries (may be top N only)
            hotkey_to_uid: Mapping from hotkey to UID (all miners in metagraph)
            
        Returns:
            Tuple of (uids, weights) - includes ALL UIDs from metagraph
        """
        # Get total number of UIDs
        max_uid = max(hotkey_to_uid.values()) if hotkey_to_uid else 0
        total_uids = max_uid + 1
        
        # Initialize weights array with zeros for all UIDs
        weights_array = [0.0] * total_uids
        
        # Create hotkey to score mapping
        hotkey_to_score = {}
        for score in scores:
            hotkey = score.get("miner_hotkey")
            if hotkey:
                hotkey_to_score[hotkey] = score.get("overall_score", 0.0)
        
        # Set weights for miners with scores
        miners_with_scores = 0
        for hotkey, uid in hotkey_to_uid.items():
            if hotkey in hotkey_to_score:
                weight = hotkey_to_score[hotkey]
                if weight > 0:
                    weights_array[uid] = weight
                    miners_with_scores += 1
        
        # Create full UID list
        uids = list(range(total_uids))
        weights = weights_array
        
        logger.info(
            f"Converted {len(scores)} scores to {total_uids} UID weights "
            f"({miners_with_scores} non-zero, {total_uids - miners_with_scores} zeros)"
        )
        
        return uids, weights
    
    async def run_once(self) -> bool:
        """
        Run one iteration of weight setting.
        
        Returns:
            True if successful, False otherwise
        """
        self.total_runs += 1
        self.last_run_at = time.time()
        
        try:
            # Step 1: Fetch scores from API
            logger.info("Step 1: Fetching latest scores...")
            scores_data = await self.fetch_latest_scores()
            if not scores_data:
                logger.warning("Failed to fetch scores, skipping this round")
                self.failed_runs += 1
                return False
            
            scores = scores_data.get("scores", [])
            
            # Step 2: Get hotkey to UID mapping
            logger.info("Step 2: Getting metagraph...")
            hotkey_to_uid = await self.get_metagraph_hotkey_to_uid()
            
            if not hotkey_to_uid:
                # Fallback to API
                logger.info("Using API fallback for hotkey to UID mapping...")
                hotkey_to_uid = await self.get_active_miners_from_api()
            
            if not hotkey_to_uid:
                logger.warning("No hotkey to UID mapping available, skipping")
                self.failed_runs += 1
                return False
            
            # Step 3: Convert scores to weights
            logger.info("Step 3: Converting scores to weights...")
            uids, weights = self.scores_to_uid_weights(scores, hotkey_to_uid)
            
            if not uids:
                logger.warning("No valid UIDs to set weights for, skipping")
                self.failed_runs += 1
                return False
            
            # Step 4: Set weights on chain
            logger.info(f"Step 4: Setting weights for {len(uids)} miners...")
            result = await self.weight_setter.set_weights(uids, weights, normalize=True)
            
            if result.success:
                logger.info(
                    f"Weights set successfully! "
                    f"Block: {result.block_number}, Attempts: {result.attempts}"
                )
                self.successful_runs += 1
                return True
            else:
                logger.error(f"Failed to set weights: {result.error_message}")
                self.failed_runs += 1
                return False
        
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
            import traceback
            traceback.print_exc()
            self.failed_runs += 1
            return False
    
    async def start(self):
        """Start the validator service loop."""
        logger.info("Starting ValidatorService...")
        self.running = True
        
        # Load wallet if not provided
        if self.wallet is None:
            try:
                from affine.core.setup import wallet
                self.wallet = wallet
                self.weight_setter.wallet = wallet
                
                if wallet:
                    logger.info(f"Loaded wallet: {wallet.hotkey.ss58_address[:16]}...")
                else:
                    logger.warning("No wallet loaded")
            except ImportError:
                logger.error("Failed to import wallet from affine.core.setup")
        
        try:
            # Get interval from environment variable
            interval = int(os.getenv("VALIDATOR_WEIGHT_SET_INTERVAL", "1800"))  # Default 30 minutes
            
            while self.running:
                # Run one iteration
                success = await self.run_once()
                
                if success:
                    logger.info(f"Next weight set in {interval}s")
                else:
                    # Use shorter interval on failure
                    retry_interval = min(interval, 300)
                    logger.info(f"Retry in {retry_interval}s after failure")
                    await asyncio.sleep(retry_interval)
                    continue
                
                # Wait for next run
                await asyncio.sleep(interval)
        
        except asyncio.CancelledError:
            logger.info("ValidatorService cancelled")
        except KeyboardInterrupt:
            logger.info("ValidatorService interrupted")
        except Exception as e:
            logger.error(f"Fatal error in validator loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            logger.info("ValidatorService stopped")
    
    async def stop(self):
        """Stop the validator service."""
        logger.info("Stopping ValidatorService...")
        self.running = False
    
    def get_metrics(self) -> Dict:
        """Get validator service metrics."""
        return {
            "service": "validator",
            "running": self.running,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": (
                self.successful_runs / self.total_runs
                if self.total_runs > 0
                else 0.0
            ),
            "last_run_at": self.last_run_at,
            "weight_setter": self.weight_setter.get_metrics(),
        }
    
    def print_status(self):
        """Print current status."""
        metrics = self.get_metrics()
        
        print("\n" + "=" * 60)
        print("Validator Service Status")
        print("=" * 60)
        print(f"Running: {metrics['running']}")
        print(f"Total Runs: {metrics['total_runs']}")
        print(f"Successful: {metrics['successful_runs']}")
        print(f"Failed: {metrics['failed_runs']}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        
        ws_metrics = metrics["weight_setter"]
        print(f"\nWeight Setter:")
        print(f"  Total Sets: {ws_metrics['total_sets']}")
        print(f"  Failed Sets: {ws_metrics['failed_sets']}")
        print(f"  Last Set: {ws_metrics['last_set_at']}")
        print("=" * 60)


async def run_service_with_mode(netuid: int, service_mode: bool):
    """Run the validator service.
    
    Args:
        netuid: Subnet UID
        service_mode: If True, run continuously; if False, run once and exit
    """
    logger.info(f"Starting validator service (netuid: {netuid})")
    
    # Create validator service
    service = ValidatorService(
        netuid=netuid,
    )
    
    # Setup signal handlers
    shutdown_event = asyncio.Event()
    
    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
    
    try:
        if not service_mode:
            # Single run mode (DEFAULT)
            logger.info("Running in one-time mode (default)")
            success = await service.run_once()
            service.print_status()
        else:
            # Continuous service mode (SERVICE_MODE=true)
            logger.info("Running in service mode (continuous). Press Ctrl+C to stop.")
            
            # Start service
            service_task = asyncio.create_task(service.start())
            
            # Wait for shutdown
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    service.print_status()
            
            # Shutdown
            await service.stop()
            service_task.cancel()
            
            try:
                await service_task
            except asyncio.CancelledError:
                pass
            
            service.print_status()
    
    except Exception as e:
        logger.error(f"Error running validator: {e}", exc_info=True)
        raise


@click.command()
@click.option(
    "--netuid",
    default=None,
    type=int,
    help="Subnet UID (default: from NETUID or 120)"
)
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(netuid, verbosity):
    """
    Affine Validator - Set weights on blockchain based on miner scores.
    
    This service fetches the latest scores from the API and sets weights
    on the Bittensor blockchain for subnet miners.
    
    Run Mode:
    - Default: One-time execution (sets weights once and exits)
    - SERVICE_MODE=true: Continuous service mode (runs every 30 minutes)
    
    Configuration:
    - NETUID: Subnet UID (default: 120)
    - SERVICE_MODE: Run as continuous service (default: false)
    """
    # Setup logging if verbosity specified
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    # Get netuid
    netuid_val = netuid if netuid is not None else int(os.getenv("NETUID", "120"))
    
    # Check service mode (default: false = one-time execution)
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    logger.info(f"Netuid: {netuid_val}")
    logger.info(f"Service mode: {service_mode}")
    
    # Run service
    asyncio.run(run_service_with_mode(
        netuid=netuid_val,
        service_mode=service_mode
    ))


if __name__ == "__main__":
    main()