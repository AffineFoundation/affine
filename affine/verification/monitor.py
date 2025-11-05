"""Monitor R2 for new incentive models."""

import os
import asyncio
from typing import Dict, Set, Optional
from pathlib import Path

from affine.storage import load_summary
from affine.setup import logger
from .queue import VerificationQueue, VerificationTask


class IncentiveMonitor:
    """Monitor R2 for new incentive models and enqueue them for verification."""

    def __init__(self, queue: VerificationQueue):
        """Initialize monitor.

        Args:
            queue: Verification queue to enqueue tasks
        """
        self.queue = queue
        self._processed_blocks: Set[int] = set()
        self._state_file = Path(
            os.getenv(
                "AFFINE_CACHE_DIR",
                Path.home() / ".cache" / "affine" / "verification",
            )
        ) / "monitor_state.txt"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load processed blocks from state file
        self._load_state()

        logger.info("Initialized incentive monitor")

    def _load_state(self) -> None:
        """Load processed blocks from state file."""
        if self._state_file.exists():
            try:
                content = self._state_file.read_text().strip()
                if content:
                    self._processed_blocks = set(int(line) for line in content.split("\n") if line.strip())
                    logger.info(f"Loaded {len(self._processed_blocks)} processed blocks from state")
            except Exception as e:
                logger.warning(f"Failed to load monitor state: {e}")

    def _save_state(self) -> None:
        """Save processed blocks to state file."""
        try:
            # Keep only last 10000 blocks to prevent file from growing too large
            recent_blocks = sorted(self._processed_blocks)[-10000:]
            self._processed_blocks = set(recent_blocks)

            content = "\n".join(str(b) for b in recent_blocks)
            self._state_file.write_text(content)
        except Exception as e:
            logger.error(f"Failed to save monitor state: {e}")

    async def check_for_new_models(self) -> int:
        """Check R2 for new incentive models.

        Returns:
            Number of new tasks enqueued
        """
        try:
            # Load latest summary from R2
            summary = await load_summary()

            # Extract block number
            block = summary.get("block")
            if block is None:
                logger.warning("No block number in summary")
                return 0

            # Check if we've already processed this block
            if block in self._processed_blocks:
                logger.debug(f"Block {block} already processed")
                return 0

            # Extract miners data
            miners_data = summary.get("data", {}).get("miners", {})
            if not miners_data:
                logger.debug(f"No miners data in block {block}")
                self._processed_blocks.add(block)
                self._save_state()
                return 0

            # Get eligible miners (those who received incentives)
            eligible_hotkeys = set()
            for hotkey, miner_info in miners_data.items():
                if miner_info.get("eligible", False):
                    eligible_hotkeys.add(hotkey)

            if not eligible_hotkeys:
                logger.info(f"No eligible miners in block {block}")
                self._processed_blocks.add(block)
                self._save_state()
                return 0

            logger.info(f"Found {len(eligible_hotkeys)} eligible miners in block {block}")

            # Enqueue verification tasks for eligible miners
            tasks_enqueued = 0
            for hotkey, miner_info in miners_data.items():
                if hotkey not in eligible_hotkeys:
                    continue

                model = miner_info.get("model")
                revision = miner_info.get("revision")
                total_score = miner_info.get("total_score", 0.0)

                if not model or not revision:
                    logger.debug(f"Skipping miner {hotkey}: missing model or revision")
                    continue

                # Calculate priority based on total score (higher score = higher priority)
                priority = int(total_score * 100)

                # Create and enqueue task
                task = VerificationTask.create(
                    block=block,
                    hotkey=hotkey,
                    model=model,
                    revision=revision,
                    priority=priority,
                )

                await self.queue.enqueue(task)
                tasks_enqueued += 1

            # Mark block as processed
            self._processed_blocks.add(block)
            self._save_state()

            logger.info(f"Enqueued {tasks_enqueued} verification tasks from block {block}")
            return tasks_enqueued

        except Exception as e:
            logger.error(f"Error checking for new models: {e}", exc_info=True)
            return 0

    async def run(self, interval_seconds: Optional[int] = None) -> None:
        """Run monitor in a loop.

        Args:
            interval_seconds: Check interval in seconds (default from env)
        """
        if interval_seconds is None:
            interval_seconds = int(os.getenv("VERIFICATION_MONITOR_INTERVAL", "300"))

        logger.info(f"Starting monitor with {interval_seconds}s interval")

        while True:
            try:
                await self.check_for_new_models()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)

            # Wait before next check
            await asyncio.sleep(interval_seconds)

    async def run_once(self) -> int:
        """Run monitor once and return.

        Returns:
            Number of new tasks enqueued
        """
        return await self.check_for_new_models()
