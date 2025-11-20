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
from affine.database.dao.system_config import SystemConfigDAO

logger = logging.getLogger(__name__)


@dataclass
class MinerInfo:
    """Miner information for task generation."""
    hotkey: str
    model_revision: str
    model: str
    chute_id: str
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
        system_config_dao: Optional[SystemConfigDAO] = None
    ):
        """
        Initialize TaskGeneratorService.
        
        Args:
            sample_results_dao: DAO for sample results
            task_queue_dao: DAO for task queue
            system_config_dao: DAO for system config
        """
        self.sample_results_dao = sample_results_dao
        self.task_queue_dao = task_queue_dao
        self.system_config_dao = system_config_dao or SystemConfigDAO()
        self._config_cache: Optional[Dict[str, Any]] = None
    
    async def _load_config_from_db(self):
        """Load configuration from SystemConfig database."""
        try:
            sampling_envs = await self.system_config_dao.get_sampling_environments()
            env_ranges = await self.system_config_dao.get_env_task_ranges()
            
            self._config_cache = {
                'sampling_envs': sampling_envs,
                'env_ranges': env_ranges
            }
            
            logger.info(
                f"Loaded config from database: {len(sampling_envs)} environments, "
                f"{len(env_ranges)} range configs"
            )
        except Exception as e:
            logger.warning(f"Failed to load config from database: {e}, using defaults")
            self._config_cache = {
                'sampling_envs': [],
                'env_ranges': {}
            }
    
    async def get_dataset_length(self, env: str) -> int:
        """Get dataset length for an environment from SystemConfig.
        
        Args:
            env: Environment name
            
        Returns:
            Dataset length (number of tasks)
            
        Raises:
            ValueError: If environment not found in SystemConfig
        """
        # Load config from database if not cached
        if self._config_cache is None:
            await self._load_config_from_db()
        
        # Get from SystemConfig
        env_ranges = self._config_cache.get('env_ranges', {})
        if env in env_ranges:
            sampling_range = env_ranges[env].get('sampling_range', [0, 0])
            if len(sampling_range) == 2:
                start, end = sampling_range
                length = end - start
                logger.debug(f"Using SystemConfig range for {env}: {start}-{end} (length={length})")
                return length
        
        # Environment not configured
        raise ValueError(
            f"Environment '{env}' not found in SystemConfig. "
            f"Please load configuration using 'python -m affine.database.cli load-config'"
        )
    
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
        dataset_length = await self.get_dataset_length(env)
        
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
                f"No missing tasks for miner {miner.uid}({miner.hotkey[:8]}...) "
                f"env={env} completed={len(completed_task_ids)} pending={len(pending_task_ids)}"
            )
            return 0
        
        logger.info(
            f"Found {len(missing_task_ids)} missing tasks for miner {miner.uid}({miner.hotkey[:8]}...) "
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
                'chute_id': miner.chute_id,
            }
            for task_id in tasks_to_create
        ]
        
        # Batch create tasks (no priority needed)
        created_count = await self.task_queue_dao.batch_create_tasks(
            tasks=task_list
        )
        
        logger.info(
            f"Created {created_count} tasks for miner {miner.uid}({miner.hotkey[:8]}...) env={env}"
        )
        
        return created_count
    
    async def generate_all_tasks(
        self,
        miners: List[MinerInfo],
        envs: Optional[List[str]] = None,
        max_tasks_per_miner_env: int = 10
    ) -> TaskGenerationResult:
        """
        Generate tasks for all miners across all environments.
        
        Args:
            miners: List of active miners
            envs: List of environments (uses SystemConfig if not provided)
            max_tasks_per_miner_env: Max tasks per miner/env combination
            
        Returns:
            TaskGenerationResult with summary
        """
        if envs is None:
            # Load config from database if not cached
            if self._config_cache is None:
                await self._load_config_from_db()
            
            # Get from SystemConfig
            sampling_envs = self._config_cache.get('sampling_envs', [])
            if not sampling_envs:
                raise ValueError(
                    "No sampling environments configured in SystemConfig. "
                    "Please load configuration using 'python -m affine.database.cli load-config'"
                )
            
            envs = sampling_envs
            logger.info(f"Using SystemConfig sampling environments: {envs}")
        
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
        retry_count = task.get('retry_count')
        max_retries = task.get('max_retries')
        
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
            envs: List of environments (uses SystemConfig if not provided)
            
        Returns:
            Dict mapping env -> completion status
        """
        if envs is None:
            # Load config from database if not cached
            if self._config_cache is None:
                await self._load_config_from_db()
            
            # Get from SystemConfig
            sampling_envs = self._config_cache.get('sampling_envs', [])
            if not sampling_envs:
                raise ValueError(
                    "No sampling environments configured in SystemConfig"
                )
            
            envs = sampling_envs
        
        status = {}
        
        for env in envs:
            dataset_length = await self.get_dataset_length(env)
            
            # Get completed task_ids
            completed_task_ids = await self.sample_results_dao.get_completed_task_ids(
                miner_hotkey=miner.hotkey,
                model_revision=miner.model_revision,
                env=env
            )
            
            # Calculate completion status
            expected_task_ids = set(range(dataset_length))
            missing_task_ids = expected_task_ids - completed_task_ids
            is_complete = len(missing_task_ids) == 0
            
            status[env] = {
                'is_complete': is_complete,
                'completed_count': len(completed_task_ids),
                'total_count': dataset_length,
                'completion_percentage': (
                    len(completed_task_ids) / dataset_length * 100
                    if dataset_length > 0 else 0
                ),
                'missing_count': len(missing_task_ids),
            }
        
        return status
    