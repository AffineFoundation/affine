#!/usr/bin/env python3
"""
Validator Service Main Entry Point

Fetches scores from backend API and sets weights on blockchain.

Usage:
    python -m affine.backend.validator.main
    python -m affine.backend.validator.main --single-run
    python -m affine.backend.validator.main --debug
"""

import asyncio
import argparse
import logging
import os
import time
import signal
import aiohttp
from typing import Dict, List, Optional, Tuple

from affine.backend.validator.weight_setter import WeightSetter
from affine.backend.config import get_config

from affine.core.setup import logger


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
        api_base_url: Optional[str] = None,
        wallet=None,
        netuid: int = 120,
    ):
        """
        Initialize ValidatorService.
        
        Args:
            api_base_url: Backend API URL
            wallet: Bittensor wallet (if None, loads from environment)
            netuid: Subnet UID
        """
        # Configuration
        self.config = get_config(api_base_url)
        self.api_base_url = api_base_url or self.config.api_base_url
        self._session: Optional[aiohttp.ClientSession] = None
        
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
        
        logger.info(f"ValidatorService initialized (API: {self.api_base_url})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session
    
    async def fetch_latest_scores(self) -> Optional[Dict]:
        """
        Fetch latest scores from backend API.
        
        Returns:
            Dictionary with scores data or None if failed
        """
        try:
            url = f"{self.api_base_url}/api/v1/scores/latest"
            session = await self._get_session()
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("No scores available from backend")
                    return None
                
                response = await resp.json()
                
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
            url = f"{self.api_base_url}/api/v1/miners/active"
            session = await self._get_session()
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    
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
        
        Args:
            scores: List of score dictionaries
            hotkey_to_uid: Mapping from hotkey to UID
            
        Returns:
            Tuple of (uids, weights)
        """
        uids = []
        weights = []
        
        for score in scores:
            hotkey = score.get("miner_hotkey")
            if not hotkey:
                continue
            
            uid = hotkey_to_uid.get(hotkey)
            if uid is None:
                logger.debug(f"Hotkey {hotkey[:16]}... not in metagraph")
                continue
            
            # Get overall score as weight
            weight = score.get("overall_score", 0.0)
            
            # Only include positive weights
            if weight > 0:
                uids.append(uid)
                weights.append(weight)
        
        logger.info(f"Converted {len(scores)} scores to {len(uids)} UID weights")
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
            while self.running:
                # Run one iteration
                success = await self.run_once()
                
                # Get interval from config
                interval = await self.config.get("validator.weight_set_interval")
                if interval is None:
                    interval = 1800  # Default 30 minutes
                
                if success:
                    logger.info(f"Next weight set in {interval}s")
                else:
                    # Use shorter interval on failure
                    interval = min(interval, 300)
                    logger.info(f"Retry in {interval}s after failure")
                
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


async def main(args):
    """Main entry point."""
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    
    # Get API URL
    api_url = args.api_url or os.getenv("API_URL") or os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Get netuid
    netuid = args.netuid or int(os.getenv("NETUID", "120"))
    
    logger.info(f"Starting validator service (API: {api_url}, netuid: {netuid})")
    
    # Create validator service
    service = ValidatorService(
        api_base_url=api_url,
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
        if args.single_run:
            # Single run mode (for debugging)
            logger.info("Running single iteration...")
            success = await service.run_once()
            service.print_status()
            return 0 if success else 1
        else:
            # Start service
            service_task = asyncio.create_task(service.start())
            
            # Wait for shutdown
            logger.info("Running in continuous mode. Press Ctrl+C to stop.")
            while not shutdown_event.is_set():
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    if args.debug:
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
        logger.error(f"Error running validator: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Affine Validator - Set weights on blockchain"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API server URL (default: from environment)"
    )
    
    parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Subnet UID (default: 1 or from environment)"
    )
    
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run once and exit (for debugging)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    exit(exit_code)