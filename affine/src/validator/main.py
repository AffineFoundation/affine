#!/usr/bin/env python3
"""
Validator Service

Fetches weights from API and sets them on-chain using Bittensor.
"""

import asyncio
import os
import click
import bittensor as bt
from typing import Dict, Optional

from affine.core.setup import logger, setup_logging
from affine.utils.api_client import create_api_client
from affine.src.config import get_config
from affine.utils.subtensor import get_subtensor
from affine.src.validator.weight_setter import WeightSetter


class ValidatorService:
    """
    Validator Service
    
    Core workflow:
    1. Wait for next weight submission window (every 180 blocks)
    2. Fetch latest weights from backend API
    3. Get burn percentage from API
    4. Set weights on chain using bittensor
    5. Verify weights were set successfully
    """
    
    def __init__(
        self,
        wallet_name: str,
        hotkey_name: str,
        netuid: int,
        network: str = "finney",
    ):
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.netuid = netuid
        self.network = network
        
        # Load wallet
        try:
            self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
            logger.info(f"Wallet: {self.wallet}")
        except Exception as e:
            logger.error(f"Failed to load wallet: {e}")
            raise

        self.api_client = None
        self.running = False
        self.weight_setter = WeightSetter(self.wallet, self.netuid)
        
    async def fetch_weights_from_api(self, max_retries: int = 12, retry_interval: int = 5) -> Optional[Dict]:
        """Fetch latest weights from backend API with retry logic
        
        Args:
            max_retries: Maximum number of retry attempts (default: 12)
            retry_interval: Seconds to wait between retries (default: 5)
        
        Returns:
            Weights data dict or None if all retries failed
        """
        if self.api_client is None:
            self.api_client = await create_api_client()
        
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.api_client.get("/scores/weights/latest")
                
                if not isinstance(response, dict) or not response.get("weights"):
                    logger.warning(f"Invalid or empty weights from API (attempt {attempt}/{max_retries})")
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_interval}s...")
                        await asyncio.sleep(retry_interval)
                        continue
                    return None
                
                weights_dict = response["weights"]
                block_number = response.get("block_number", "unknown")
                
                if attempt > 1:
                    logger.info(f"Successfully fetched weights on attempt {attempt}/{max_retries}")
                logger.info(f"Fetched {len(weights_dict)} weights (block={block_number})")
                return response
            
            except Exception as e:
                logger.error(f"Error fetching weights (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_interval}s...")
                    await asyncio.sleep(retry_interval)
                else:
                    logger.error(f"Failed to fetch weights after {max_retries} attempts")
                    return None
        
        return None

    async def wait_for_next_window(self, subtensor, interval_blocks: int):
        """Wait for the next weight submission window"""
        current_block = await subtensor.get_current_block()
        
        # Calculate next window
        current_epoch = current_block // interval_blocks
        next_window_start = (current_epoch + 1) * interval_blocks
        blocks_remaining = next_window_start - current_block
        
        if blocks_remaining <= 0:
            logger.info(f"In submission window at block {current_block}")
            return current_block
        
        logger.info(f"Waiting for block {next_window_start} ({blocks_remaining} blocks remaining)")
        
        # Wait block by block until we reach the target
        while self.running and current_block < next_window_start:
            current_block = await subtensor.get_current_block()
            blocks_remaining = next_window_start - current_block
            
            if blocks_remaining > 0:
                logger.info(f"Current block: {current_block}, target: {next_window_start}, remaining: {blocks_remaining}")
                await subtensor.wait_for_block(current_block + 1)
            else:
                break
        
        return next_window_start

    async def run_iteration(self):
        """Run one iteration of weight setting"""
        # 1. Fetch weights
        weights_data = await self.fetch_weights_from_api()
        if not weights_data:
            return

        # 2. Get config
        config = get_config()
        burn_percentage = await config.get("validator_burn_percentage") or 0.0
        burn_percentage = float(burn_percentage)

        # 3. Set weights using WeightSetter
        await self.weight_setter.set_weights(
            weights_data.get("weights", {}),
            burn_percentage
        )

    async def start(self):
        """Start the validator service"""
        logger.info("Starting ValidatorService...")
        self.running = True
        
        config = get_config()
        
        while self.running:
            try:
                interval_blocks = await config.get("weight_set_interval_blocks")
                if interval_blocks is None:
                    interval_blocks = int(os.getenv("WEIGHT_SET_INTERVAL_BLOCKS", "180"))
                else:
                    interval_blocks = int(interval_blocks)

                subtensor = await get_subtensor()
                await self.wait_for_next_window(subtensor, interval_blocks)
                
                await self.run_iteration()
                
                # Sleep a bit to avoid tight loop if something goes wrong with window calculation
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        self.running = False
        logger.info("Stopping ValidatorService...")


async def run_service(wallet_name, hotkey_name, netuid, network):
    service = ValidatorService(wallet_name, hotkey_name, netuid, network)
    task = asyncio.create_task(service.start())
    try:
        await task
    except asyncio.CancelledError:
        pass

@click.command()
@click.option("--netuid", type=int, default=120)
@click.option("--wallet-name", type=str, default=os.getenv("BT_WALLET_COLD"))
@click.option("--hotkey-name", type=str, default=os.getenv("BT_WALLET_HOT"))
@click.option("--network", type=str, default="finney")
@click.option("--verbosity", type=str, default="1")
def main(netuid, wallet_name, hotkey_name, network, verbosity):
    setup_logging(int(verbosity))
    
    if not wallet_name or not hotkey_name:
        logger.error("Wallet name and hotkey name are required")
        return

    asyncio.run(run_service(wallet_name, hotkey_name, netuid, network))

if __name__ == "__main__":
    main()