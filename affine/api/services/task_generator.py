"""
Task Generator Service

Automatically generates sampling tasks for all miners across all environments.
Ensures complete dataset coverage by detecting missing task_ids.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.task_queue import TaskQueueDAO

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset/environment."""
    name: str
    length: int
    priority: int = 1000


# Default dataset configurations
# TODO: Load from SystemConfigDAO or environment
DEFAULT_DATASETS = {
    "affine:sat": DatasetConfig("affine:sat", 200, 1000),
    "affine:abd": DatasetConfig("affine:abd", 200, 1000),
    "affine:ded": DatasetConfig("affine:ded", 200, 1000),
    "agentgym:webshop": DatasetConfig("agentgym:webshop", 100, 900),
    "agentgym:wordarena": DatasetConfig("agentgym:wordarena", 100, 900),
}


@dataclass
class MinerInfo:
    """Miner information for task generation."""
    hotkey: str
    model_revision: str
    model: str
    uid: int = -1


@dataclass
class TaskGenerationResult:
    """Result of task generation operation."""
    total_tasks_created: int
    tasks_by_env: Dict[str, int]
    miners_processed: int
    errors: List[str]


class TaskGeneratorService:
    """
    Service for generating sampling tasks.
    
    Responsibilities:
    1. Query all active miners from chain/cache
    2. For each miner+env combination, find missing task_ids
    3. Generate tasks for missing task_ids
    4. Clean up invalid tasks (miners no longer active)
    """
    
    def __init__(
        self,
        sample_results_dao: SampleResultsDAO,
        task_queue_dao: TaskQueueDAO,
        datasets: Optional[Dict[str, DatasetConfig]] = None
    ):
        """
        Initialize TaskGeneratorService.
        
        Args:
            sample_results_dao: DAO for sample results
            task_queue_dao: DAO for task queue
            datasets: Dataset configurations (uses defaults if not provided)
        """
        self.sample_results_dao = sample_results_dao
        self.task_queue_dao = task_queue_dao
        self.datasets = datasets or DEFAULT_DATASETS
    
    def get_dataset_length(self, env: str) -> int:
        """Get dataset length for an environment.
        
        Args:
            env: Environment name
            
        Returns:
            Dataset length (number of tasks)
        """
        if env in self.datasets:
            return self.datasets[env].length
        
        # Default fallback
        logger.warning(f"Unknown environment {env}, using default length 200")
        return 200
    
    def get_dataset_priority(self, env: str) -> int:
        """Get priority for an environment.
        
        Args:
            env: Environment name
            
        Returns:
            Priority value
        """
        if env in self.datasets:
            return self.datasets[env].priority
        
        return 1000  # Default priority
    
    async def generate_tasks_for_miner_env(
        self,
        miner: MinerInfo,
        env: str,
        max_tasks_per_batch: int = 100
    ) -> int:
        """
        Generate missing tasks for a specific miner and environment.
        
        Args:
            miner: Miner information
            env: Environment name
            max_tasks_per_batch: Maximum tasks to create in one batch
            
        Returns:
            Number of tasks created
        """
        dataset_length = self.get_dataset_length(env)
        priority = self.get_dataset_priority(env)
        
        # Get expected task_ids (all indices in dataset)
        expected_task_ids = set(range(dataset_length))
        
        # Get completed task_ids from sample results
        completed_task_ids = await self.sample_results_dao.get_completed_task_ids(
            miner_hotkey=miner.hotkey,
            model_revision=miner.model_revision,
            env=env
        )
        
        # Get pending task_ids already in queue
        pending_task_ids = await self.task_queue_dao.get_pending_task_ids_for_miner(
            miner_hotkey=miner.hotkey,
            model_revision=miner.model_revision,
            env=env
        )
        
        # Calculate missing task_ids
        missing_task_ids = expected_task_ids - completed_task_ids - pending_task_ids
        
        if not missing_task_ids:
            logger.debug(
                f"No missing tasks for miner {miner.hotkey[:8]}... "
                f"env={env} completed={len(completed_task_ids)} pending={len(pending_task_ids)}"
            )
            return 0
        
        logger.info(
            f"Found {len(missing_task_ids)} missing tasks for miner {miner.hotkey[:8]}... "
            f"env={env} (completed={len(completed_task_ids)}, pending={len(pending_task_ids)})"
        )
        
        # Limit batch size
        tasks_to_create = sorted(missing_task_ids)[:max_tasks_per_batch]
        
        # Prepare task data
        task_list = [
            {
                'miner_hotkey': miner.hotkey,
                'model_revision': miner.model_revision,
                'model': miner.model,
                'env': env,
                'task_id': task_id,
            }
            for task_id in tasks_to_create
        ]
        
        # Batch create tasks
        created_count = await self.task_queue_dao.batch_create_tasks(
            tasks=task_list,
            priority=priority
        )
        
        logger.info(
            f"Created {created_count} tasks for miner {miner.hotkey[:8]}... env={env}"
        )
        
        return created_count
    
    async def generate_all_tasks(
        self,
        miners: List[MinerInfo],
        envs: Optional[List[str]] = None,
        max_tasks_per_miner_env: int = 100
    ) -> TaskGenerationResult:
        """
        Generate tasks for all miners across all environments.
        
        Args:
            miners: List of active miners
            envs: List of environments (uses all configured if not provided)
            max_tasks_per_miner_env: Max tasks per miner/env combination
            
        Returns:
            TaskGenerationResult with summary
        """
        if envs is None:
            envs = list(self.datasets.keys())
        
        result = TaskGenerationResult(
            total_tasks_created=0,
            tasks_by_env={env: 0 for env in envs},
            miners_processed=len(miners),
            errors=[]
        )
        
        logger.info(
            f"Starting task generation for {len(miners)} miners "
            f"across {len(envs)} environments"
        )
        
        for miner in miners:
            for env in envs:
                try:
                    created = await self.generate_tasks_for_miner_env(
                        miner=miner,
                        env=env,
                        max_tasks_per_batch=max_tasks_per_miner_env
                    )
                    result.total_tasks_created += created
                    result.tasks_by_env[env] += created
                    
                except Exception as e:
                    error_msg = (
                        f"Error generating tasks for miner {miner.hotkey[:8]}... "
                        f"env={env}: {str(e)}"
                    )
                    logger.error(error_msg)
                    result.errors.append(error_msg)
        
        logger.info(
            f"Task generation complete: created {result.total_tasks_created} tasks, "
            f"{len(result.errors)} errors"
        )
        
        return result
    
    async def cleanup_invalid_tasks(
        self,
        valid_miners: List[MinerInfo]
    ) -> int:
        """
        Remove tasks for miners that are no longer valid.
        
        This is called periodically to ensure the queue only contains
        tasks for currently active miners.
        
        Args:
            valid_miners: List of currently valid miners
            
        Returns:
            Number of tasks removed
        """
        # Convert to format expected by DAO
        valid_miner_dicts = [
            {'hotkey': m.hotkey, 'model_revision': m.model_revision}
            for m in valid_miners
        ]
        
        removed_count = await self.task_queue_dao.cleanup_invalid_tasks(valid_miner_dicts)
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} invalid tasks")
        
        return removed_count
    
    async def validate_task_for_execution(
        self,
        task: Dict[str, Any],
        valid_miners: List[MinerInfo]
    ) -> bool:
        """
        Validate that a task is still valid for execution.
        
        Checks:
        1. Miner hotkey + model_revision is in valid_miners
        2. Task hasn't exceeded retry limit
        
        Args:
            task: Task dict from queue
            valid_miners: List of valid miners
            
        Returns:
            True if task is valid, False otherwise
        """
        task_hotkey = task.get('miner_hotkey')
        task_revision = task.get('model_revision')
        
        # Check if miner is still valid
        is_valid = any(
            m.hotkey == task_hotkey and m.model_revision == task_revision
            for m in valid_miners
        )
        
        if not is_valid:
            logger.warning(
                f"Task {task.get('task_uuid')} for miner {task_hotkey[:8]}... "
                f"is no longer valid (miner not in active list)"
            )
            return False
        
        # Check retry count
        retry_count = task.get('retry_count', 0)
        max_retries = task.get('max_retries', 3)
        
        if retry_count >= max_retries:
            logger.warning(
                f"Task {task.get('task_uuid')} exceeded max retries ({retry_count}/{max_retries})"
            )
            return False
        
        return True
    
    async def get_completion_status_for_miner(
        self,
        miner: MinerInfo,
        envs: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get sampling completion status for a miner across all environments.
        
        Args:
            miner: Miner information
            envs: List of environments (uses all configured if not provided)
            
        Returns:
            Dict mapping env -> completion status
        """
        if envs is None:
            envs = list(self.datasets.keys())
        
        status = {}
        
        for env in envs:
            dataset_length = self.get_dataset_length(env)
            
            result = await self.sample_results_dao.get_samples_with_completion_status(
                miner_hotkey=miner.hotkey,
                model_revision=miner.model_revision,
                env=env,
                dataset_length=dataset_length,
                deduplicate_by_task_id=True,
                include_extra=False
            )
            
            status[env] = {
                'is_complete': result['is_complete'],
                'completed_count': result['completed_count'],
                'total_count': result['total_count'],
                'completion_percentage': (
                    result['completed_count'] / result['total_count'] * 100
                    if result['total_count'] > 0 else 0
                ),
                'missing_count': len(result['missing_task_ids']),
            }
        
        return status
    
    def update_dataset_config(self, env: str, length: int, priority: int = 1000):
        """
        Update dataset configuration dynamically.
        
        Args:
            env: Environment name
            length: Dataset length
            priority: Task priority
        """
        self.datasets[env] = DatasetConfig(env, length, priority)
        logger.info(f"Updated dataset config: {env} -> length={length}, priority={priority}")