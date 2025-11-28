#!/usr/bin/env python3
"""
Simplified Validator Service

Fetches weights from API and sets them on-chain.
No longer depends on fiber or bittensor libraries except for chain interaction.
"""

import asyncio
import os
import time
import signal
import click
from typing import Dict, List, Optional

from affine.src.validator.weight_setter import WeightSetter
from affine.core.setup import logger, setup_logging
from affine.utils.api_client import create_api_client
from affine.src.validator.chain import get_substrate, query_chain


class ValidatorService:
    """
    Simplified Validator Service
    
    Core workflow:
    1. Fetch latest weights from backend API
    2. Get burn percentage from API
    3. Set weights on chain (with normalization and burn applied)
    
    The API provides normalized weights that are ready to use.
    The validator just needs to:
    - Apply burn mechanism (allocate % to UID 0)
    - Convert to chain format
    - Submit to blockchain
    """
    
    def __init__(
        self,
        wallet_name: str,
        hotkey_name: str,
        netuid: int,
        network: str = "finney",
        network_address: Optional[str] = None,
    ):
        """
        Initialize validator service
        
        Args:
            wallet_name: Wallet name
            hotkey_name: Hotkey name
            netuid: Subnet UID
            network: Network name (finney, test, local)
            network_address: Custom network address
        """
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = netuid
        self.network = network
        self.network_address = network_address
        
        # API client
        self.api_client = create_api_client()
        
        # Weight setter
        self.weight_setter = WeightSetter(
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
            netuid=netuid,
            network=network,
            network_address=network_address,
            max_retries=3,
            retry_delay=10.0,
        )
        
        # State
        self.running = False
        self.last_run_at: Optional[float] = None
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        
        logger.debug(f"ValidatorService initialized for {wallet_name}/{hotkey_name}")
    
    async def fetch_weights_from_api(self) -> Optional[Dict]:
        """
        Fetch latest weights from backend API with comprehensive validation
        
        Returns:
            Dictionary with weights data or None if failed
        """
        try:
            response = await self.api_client.get("/scores/weights/latest")
            
            # Validate response format
            if not isinstance(response, dict):
                logger.error(f"Invalid API response format: expected dict, got {type(response)}")
                return None
            
            weights_dict = response.get("weights")
            if not weights_dict:
                logger.warning("No weights available from API")
                return None
            
            if not isinstance(weights_dict, dict):
                logger.error(f"Invalid weights format: expected dict, got {type(weights_dict)}")
                return None
            
            # Validate weight count
            if len(weights_dict) == 0:
                logger.warning("Empty weights dictionary from API")
                return None
            
            # Validate block_number exists
            block_number = response.get("block_number")
            if block_number is None:
                logger.warning("No block_number in API response")
            
            # Validate weight values
            total_weight = 0.0
            valid_count = 0
            for uid_str, weight_data in weights_dict.items():
                try:
                    # Validate UID format
                    uid = int(uid_str)
                    if uid < 0:
                        logger.warning(f"Negative UID in API response: {uid_str}")
                        continue
                    
                    # Validate weight data
                    if not isinstance(weight_data, dict):
                        logger.warning(f"Invalid weight data for UID {uid_str}: {type(weight_data)}")
                        continue
                    
                    weight = weight_data.get("weight")
                    if weight is None:
                        logger.warning(f"Missing weight value for UID {uid_str}")
                        continue
                    
                    try:
                        weight_float = float(weight)
                        if weight_float < 0:
                            logger.warning(f"Negative weight for UID {uid_str}: {weight_float}")
                            continue
                        if weight_float > 1.0:
                            logger.warning(f"Weight > 1.0 for UID {uid_str}: {weight_float}")
                        
                        total_weight += weight_float
                        valid_count += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid weight value for UID {uid_str}: {weight} ({e})")
                        continue
                
                except ValueError:
                    logger.warning(f"Invalid UID format: {uid_str}")
                    continue
            
            if valid_count == 0:
                logger.error("No valid weights found in API response")
                return None
            
            # Validate total weight sum (should be close to 1.0 if pre-normalized)
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(
                    f"Unexpected weight sum: {total_weight:.6f} "
                    f"(expected ~1.0 for {valid_count} weights)"
                )
            
            logger.info(
                f"Fetched {len(weights_dict)} weights from API "
                f"({valid_count} valid, sum={total_weight:.6f}, block={block_number})"
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error fetching weights: {e}")
            return None
    
    async def convert_api_weights_to_chain_format(
        self,
        api_weights: Dict[str, Dict]
    ) -> tuple[List[int], List[float]]:
        """
        Convert API weights (UID-based dictionary) to chain format (lists).
        
        Filters out zero/negative weights and validates UIDs during conversion.
        
        Args:
            api_weights: Weights from API {uid_str: {weight: float, ...}}
            
        Returns:
            Tuple of (uids_list, weights_list) with only positive weights
        """
        uids = []
        weights = []
        
        for uid_str, weight_data in api_weights.items():
            try:
                # Validate and parse UID
                uid = int(uid_str)
                if uid < 0:
                    logger.warning(f"Skipping negative UID: {uid_str}")
                    continue
                
                # Validate weight data structure
                if not isinstance(weight_data, dict):
                    logger.warning(f"Skipping UID {uid_str}: invalid data type {type(weight_data)}")
                    continue
                
                # Extract and validate weight value
                weight_value = weight_data.get("weight")
                if weight_value is None:
                    logger.warning(f"Skipping UID {uid_str}: missing weight value")
                    continue
                
                try:
                    weight = float(weight_value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping UID {uid_str}: invalid weight value {weight_value} ({e})")
                    continue
                
                # Only include positive weights (filter out zero and negative)
                if weight > 0:
                    uids.append(uid)
                    weights.append(weight)
                elif weight < 0:
                    logger.warning(f"Skipping UID {uid_str}: negative weight {weight}")
                # weight == 0 is silently skipped (normal case)
            
            except ValueError as e:
                logger.warning(f"Skipping invalid UID format '{uid_str}': {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing UID {uid_str}: {e}")
                continue
        
        if len(uids) == 0:
            logger.error("No valid positive weights found after conversion")
        else:
            weight_sum = sum(weights)
            logger.info(
                f"Converted {len(uids)} positive weights "
                f"(sum: {weight_sum:.6f}, min: {min(weights):.6f}, max: {max(weights):.6f})"
            )
        
        return uids, weights
    
    async def run_once(self) -> bool:
        """
        Run one iteration of weight setting
        
        Returns:
            True if successful, False otherwise
        """
        self.total_runs += 1
        self.last_run_at = time.time()
        
        try:
            # Fetch weights from API
            weights_data = await self.fetch_weights_from_api()
            
            if not weights_data:
                logger.warning("No weights available")
                self.failed_runs += 1
                return False
            
            api_weights = weights_data.get("weights", {})
            block_number = weights_data.get("block_number")
            
            # Convert weights (API already returns UID-based weights)
            uids, weights = await self.convert_api_weights_to_chain_format(api_weights)
            
            if not uids:
                logger.warning("No valid weights after conversion")
                self.failed_runs += 1
                return False
            
            logger.info(f"Converted {len(uids)} weights for block {block_number}")
            
            # Get burn percentage
            try:
                burn_config = await self.api_client.get("/config/validator_burn_percentage")
                burn_percentage = float(burn_config.get("param_value", 0.0))
                if burn_percentage > 0:
                    logger.info(f"Burn: {burn_percentage:.1%}")
            except Exception as e:
                logger.debug(f"Using default burn 0.0: {e}")
                burn_percentage = 0.0
            
            # Set weights on chain
            result = await self.weight_setter.set_weights(
                uids=uids,
                weights=weights,
                burn_percentage=burn_percentage,
            )
            
            if result.success:
                self.successful_runs += 1
                return True
            else:
                logger.error(f"Failed: {result.error_message}")
                self.failed_runs += 1
                return False
        
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
            import traceback
            traceback.print_exc()
            self.failed_runs += 1
            return False
    
    async def start(self):
        """Start the validator service loop"""
        logger.info("Starting ValidatorService...")
        self.running = True
        
        try:
            # Get interval from environment
            interval = int(os.getenv("VALIDATOR_WEIGHT_SET_INTERVAL", "1800"))  # Default 30 minutes
            
            while self.running:
                # Run one iteration
                success = await self.run_once()
                
                if success:
                    logger.info(f"Next weight set in {interval}s")
                    await asyncio.sleep(interval)
                else:
                    # Use shorter interval on failure
                    retry_interval = min(interval, 300)
                    logger.info(f"Retry in {retry_interval}s after failure")
                    await asyncio.sleep(retry_interval)
        
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
        """Stop the validator service"""
        logger.info("Stopping ValidatorService...")
        self.running = False
    
    def get_metrics(self) -> Dict:
        """Get validator service metrics"""
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
        """Print current status"""
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


async def run_service_with_mode(
    wallet_name: str,
    hotkey_name: str,
    netuid: int,
    network: str,
    service_mode: bool
):
    """Run the validator service"""
    logger.info(f"Starting validator (netuid: {netuid}, network: {network})")
    
    service = ValidatorService(
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        netuid=netuid,
        network=network,
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
            # Continuous service mode
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
    "--wallet-name",
    default=None,
    type=str,
    help="Wallet name (default: from BT_WALLET_COLD)"
)
@click.option(
    "--hotkey-name",
    default=None,
    type=str,
    help="Hotkey name (default: from BT_WALLET_HOT)"
)
@click.option(
    "--network",
    default=None,
    type=str,
    help="Network (default: from SUBTENSOR_NETWORK or finney)"
)
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
def main(netuid, wallet_name, hotkey_name, network, verbosity):
    """Affine Validator - Set weights on blockchain"""
    if verbosity is not None:
        setup_logging(int(verbosity))
    
    wallet_name_val = wallet_name or os.getenv("BT_WALLET_COLD")
    hotkey_name_val = hotkey_name or os.getenv("BT_WALLET_HOT")
    netuid_val = netuid if netuid is not None else int(os.getenv("NETUID", "120"))
    network_val = network or os.getenv("SUBTENSOR_NETWORK", "finney")
    service_mode = os.getenv("SERVICE_MODE", "false").lower() in ("true", "1", "yes")
    
    if not wallet_name_val:
        raise click.UsageError("Wallet name required (--wallet-name or BT_WALLET_COLD)")
    if not hotkey_name_val:
        raise click.UsageError("Hotkey name required (--hotkey-name or BT_WALLET_HOT)")
    
    logger.info(f"Wallet: {wallet_name_val}/{hotkey_name_val}")
    logger.info(f"Network: {network_val}")
    logger.info(f"Netuid: {netuid_val}")
    logger.info(f"Service mode: {service_mode}")
    
    asyncio.run(run_service_with_mode(
        wallet_name=wallet_name_val,
        hotkey_name=hotkey_name_val,
        netuid=netuid_val,
        network=network_val,
        service_mode=service_mode
    ))


if __name__ == "__main__":
    main()