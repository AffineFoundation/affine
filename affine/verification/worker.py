"""Worker for processing model verification tasks."""

import os
import asyncio
from typing import Optional

from affine.setup import logger
from .queue import VerificationQueue, VerificationTask
from .blacklist import BlacklistManager
from .deployment import ModelDeployment, DeploymentInfo
from .similarity import SimilarityChecker, get_sample_prompts


class VerificationWorker:
    """Worker for processing verification tasks."""

    def __init__(
        self,
        queue: VerificationQueue,
        blacklist_manager: BlacklistManager,
        deployment: ModelDeployment,
        similarity_checker: SimilarityChecker,
    ):
        """Initialize worker.

        Args:
            queue: Verification queue
            blacklist_manager: Blacklist manager
            deployment: Model deployment manager
            similarity_checker: Similarity checker
        """
        self.queue = queue
        self.blacklist_manager = blacklist_manager
        self.deployment = deployment
        self.similarity_checker = similarity_checker

        # Configuration
        self.sample_size = int(os.getenv("VERIFICATION_SAMPLE_SIZE", "10"))
        self.similarity_threshold = float(os.getenv("VERIFICATION_SIMILARITY_THRESHOLD", "0.85"))

        logger.info(
            f"Initialized verification worker "
            f"(sample_size={self.sample_size}, threshold={self.similarity_threshold})"
        )

    async def process_task(self, task: VerificationTask) -> bool:
        """Process a single verification task.

        Args:
            task: Verification task to process

        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"Processing task {task.task_id}: model={task.model}, hotkey={task.hotkey}")

        deployment_info: Optional[DeploymentInfo] = None

        try:
            # Step 1: Deploy model to GPU machine
            logger.info(f"Deploying model {task.model}@{task.revision}")
            deployment_info = await self.deployment.deploy_model(
                model=task.model,
                revision=task.revision,
            )

            # Step 2: Get sample prompts
            logger.info(f"Getting {self.sample_size} sample prompts")
            prompts = await get_sample_prompts(
                hotkey=task.hotkey,
                block=task.block,
                sample_size=self.sample_size,
            )

            if not prompts:
                raise ValueError("No prompts available for testing")

            logger.info(f"Got {len(prompts)} prompts for testing")

            # Step 3: Build Chutes endpoint
            # Get slug from miner info
            from affine.miners import miners as get_miners

            miners_dict = await get_miners(check_validity=False)
            miner = None

            for m in miners_dict.values():
                if m.hotkey == task.hotkey:
                    miner = m
                    break

            if miner is None or not miner.slug:
                raise ValueError(f"Miner slug not found for hotkey {task.hotkey}")

            chutes_endpoint = f"https://{miner.slug}.chutes.ai/v1"
            logger.info(f"Chutes endpoint: {chutes_endpoint}")
            logger.info(f"Local endpoint: {deployment_info.endpoint}")

            # Step 4: Compare outputs
            logger.info("Comparing outputs between Chutes and local deployment")
            comparison_results = await self.similarity_checker.compare_outputs(
                chutes_endpoint=chutes_endpoint,
                local_endpoint=deployment_info.endpoint,
                model=task.model,
                prompts=prompts,
            )

            # Step 5: Calculate average similarity
            valid_results = [r for r in comparison_results if r.error is None]
            if not valid_results:
                raise ValueError("No valid comparison results")

            avg_similarity = sum(r.similarity_score for r in valid_results) / len(valid_results)
            logger.info(
                f"Average similarity: {avg_similarity:.4f} "
                f"({len(valid_results)}/{len(comparison_results)} valid results)"
            )

            # Step 6: Check threshold and update blacklist if needed
            if avg_similarity < self.similarity_threshold:
                logger.warning(
                    f"Similarity {avg_similarity:.4f} below threshold {self.similarity_threshold}, "
                    f"adding to blacklist"
                )

                # Collect details
                details = {
                    "avg_similarity": avg_similarity,
                    "valid_results": len(valid_results),
                    "total_results": len(comparison_results),
                    "samples": [
                        {
                            "prompt": r.prompt[:100],
                            "similarity": r.similarity_score,
                            "chutes_output": r.chutes_output[:200],
                            "local_output": r.local_output[:200],
                        }
                        for r in valid_results[:3]  # Keep first 3 samples
                    ],
                }

                await self.blacklist_manager.add_to_blacklist(
                    hotkey=task.hotkey,
                    model=task.model,
                    reason="low_similarity",
                    similarity_score=avg_similarity,
                    block=task.block,
                    samples_tested=len(valid_results),
                    details=details,
                )

                logger.info(f"Added {task.hotkey} to blacklist")
            else:
                logger.info(f"Similarity {avg_similarity:.4f} OK, no blacklist action needed")

            # Mark task as completed
            await self.queue.complete_task(task.task_id)

            return True

        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}", exc_info=True)

            # Mark task as failed
            await self.queue.fail_task(task.task_id, str(e), retry=True)

            return False

        finally:
            # Cleanup deployment
            if deployment_info:
                try:
                    logger.info(f"Cleaning up deployment {deployment_info.container_id[:12]}")
                    await self.deployment.cleanup_deployment(deployment_info.container_id)
                except Exception as e:
                    logger.error(f"Error cleaning up deployment: {e}")

    async def run(self, max_tasks: Optional[int] = None) -> None:
        """Run worker in a loop.

        Args:
            max_tasks: Maximum number of tasks to process (None = infinite)
        """
        logger.info(f"Starting worker (max_tasks={max_tasks})")

        tasks_processed = 0

        while True:
            # Check if we've reached max tasks
            if max_tasks is not None and tasks_processed >= max_tasks:
                logger.info(f"Reached max tasks ({max_tasks}), stopping")
                break

            # Get next task
            task = await self.queue.dequeue()

            if task is None:
                # No tasks available, wait and retry
                logger.debug("No tasks available, waiting...")
                await asyncio.sleep(10)
                continue

            # Process task
            success = await self.process_task(task)

            if success:
                tasks_processed += 1
                logger.info(f"Processed task {task.task_id} successfully ({tasks_processed} total)")
            else:
                logger.warning(f"Failed to process task {task.task_id}")

        logger.info("Worker stopped")

    async def run_once(self) -> bool:
        """Process one task and return.

        Returns:
            True if a task was processed, False if queue was empty
        """
        task = await self.queue.dequeue()

        if task is None:
            logger.info("No tasks available")
            return False

        success = await self.process_task(task)
        return success
