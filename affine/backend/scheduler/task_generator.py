"""
Task Generator Module

Generates batch sampling tasks by identifying missing task_ids based on:
1. Dataset range (0 to dataset_length-1)
2. Already sampled task_ids in results
3. Already queued task_ids in task pool
"""

from typing import Dict, List, Set
from affine.backend.config import get_config
from affine.backend.scheduler.miner_discovery import MinerInfo
from affine.core.http_client import AsyncHTTPClient
from affine.core.setup import logger, get_env_names


class TaskGenerator:
    """Generates batch sampling tasks by identifying missing task_ids."""
    
    def __init__(self, api_base_url: str = None):
        """Initialize task generator.
        
        Args:
            api_base_url: API server URL (default: from config)
        """
        self.config = get_config(api_base_url)
        self.api_base_url = api_base_url or self.config.api_base_url
        self.http_client = AsyncHTTPClient(timeout=30)
    
    async def _get_dataset_length(self, env: str) -> int:
        """Get dataset length for environment.
        
        Args:
            env: Environment name (e.g., 'affine:sat')
        
        Returns:
            Dataset length
        """
        try:
            url = f"{self.api_base_url}/api/v1/config/env/{env}/dataset-length"
            response = await self.http_client.get(url)
            
            if response:
                return response.get("dataset_length", 0)
            else:
                logger.warning(f"[TaskGenerator] Failed to get dataset length for {env}")
                return 0
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error getting dataset length: {e}")
            return 0
    
    async def _get_sampled_task_ids(
        self, miner: MinerInfo, env: str
    ) -> Set[int]:
        """Get already sampled task_ids from results.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            Set of sampled task_ids
        """
        try:
            url = f"{self.api_base_url}/api/v1/samples/task-ids"
            params = {
                "miner_hotkey": miner.hotkey,
                "model_revision": miner.revision,
                "env": env
            }
            response = await self.http_client.get(url, params=params)
            
            if response:
                return set(response.get("task_ids", []))
            else:
                logger.warning(
                    f"[TaskGenerator] Failed to get sampled task_ids for U{miner.uid} {env}"
                )
                return set()
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error getting sampled task_ids: {e}")
            return set()
    
    async def _get_queued_task_ids(
        self, miner: MinerInfo, env: str
    ) -> Set[int]:
        """Get already queued task_ids from task pool.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            Set of queued task_ids
        """
        try:
            url = f"{self.api_base_url}/api/v1/tasks/queued-task-ids"
            params = {
                "miner_hotkey": miner.hotkey,
                "model_revision": miner.revision,
                "env": env
            }
            response = await self.http_client.get(url, params=params)
            
            if response:
                return set(response.get("task_ids", []))
            else:
                logger.warning(
                    f"[TaskGenerator] Failed to get queued task_ids for U{miner.uid} {env}"
                )
                return set()
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error getting queued task_ids: {e}")
            return set()
    
    async def generate_missing_tasks(
        self, miner: MinerInfo, env: str
    ) -> List[Dict]:
        """Generate tasks for missing task_ids in the dataset range.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            List of task dictionaries for missing task_ids
        """
        # Get dataset range
        dataset_length = await self._get_dataset_length(env)
        if dataset_length == 0:
            logger.warning(f"[TaskGenerator] Dataset length is 0 for {env}")
            return []
        
        # Get already sampled and queued task_ids
        sampled_ids = await self._get_sampled_task_ids(miner, env)
        queued_ids = await self._get_queued_task_ids(miner, env)
        
        # Calculate full range
        full_range = set(range(dataset_length))
        
        # Find missing task_ids
        existing_ids = sampled_ids | queued_ids
        missing_ids = full_range - existing_ids
        
        logger.info(
            f"[TaskGenerator] U{miner.uid} {env}: "
            f"dataset={dataset_length}, sampled={len(sampled_ids)}, "
            f"queued={len(queued_ids)}, missing={len(missing_ids)}"
        )
        
        # Generate tasks for missing task_ids
        tasks = []
        for task_id in sorted(missing_ids):
            task = {
                "miner_hotkey": miner.hotkey,
                "model_revision": miner.revision,
                "model": miner.model,
                "env": env,
                "task_id": task_id,
                "priority": 0,
            }
            tasks.append(task)
        
        return tasks
    
    async def generate_tasks_for_miner(self, miner: MinerInfo) -> List[Dict]:
        """Generate all missing tasks for a miner across all environments.
        
        Args:
            miner: Miner information
        
        Returns:
            List of task dictionaries (empty list if all complete)
        """
        all_tasks = []
        envs = get_env_names()
        
        for env in envs:
            tasks = await self.generate_missing_tasks(miner, env)
            all_tasks.extend(tasks)
        
        logger.info(
            f"[TaskGenerator] Generated {len(all_tasks)} tasks for U{miner.uid}"
        )
        
        return all_tasks
    
    async def submit_tasks_to_pool(self, tasks: List[Dict]) -> int:
        """Submit generated tasks to the task pool via API.
        
        Args:
            tasks: List of task dictionaries
        
        Returns:
            Number of tasks successfully submitted
        """
        if not tasks:
            return 0
        
        try:
            url = f"{self.api_base_url}/api/v1/tasks/batch"
            response = await self.http_client.post(url, json={"tasks": tasks})
            
            if response:
                count = response.get("added_count", 0)
                logger.info(f"[TaskGenerator] Submitted {count} tasks to pool")
                return count
            else:
                logger.error("[TaskGenerator] Failed to submit tasks to pool")
                return 0
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error submitting tasks: {e}")
            return 0