"""
Task Generator Module

Generates batch sampling tasks by identifying missing task_ids based on:
1. Sampling environment configuration (from SystemConfig or env vars)
2. Task ID range per environment (sampling_range)
3. Already sampled task_ids in results
4. Already queued task_ids in task pool
"""

import os
import json
import aiohttp
from typing import Dict, List, Set, Tuple, Optional
from affine.backend.config import get_config
from affine.api.services.miners_cache import MinerInfo
from affine.core.setup import logger


class TaskGenerator:
    """Generates batch sampling tasks by identifying missing task_ids.
    
    Configuration Priority:
    1. Environment variables (SAMPLING_ENVIRONMENTS, ENV_TASK_RANGES)
    2. SystemConfig database values
    """
    
    def __init__(self, api_base_url: str = None):
        """Initialize task generator.
        
        Args:
            api_base_url: API server URL (default: from config)
        """
        self.config = get_config(api_base_url)
        self.api_base_url = api_base_url or self.config.api_base_url
        self._env_config_cache = None
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def _get_sampling_environments(self) -> List[str]:
        """Get list of sampling environments with priority: env vars > SystemConfig.
        
        Returns:
            List of environment names for sampling
        """
        # Priority 1: Environment variable
        env_var = os.getenv("SAMPLING_ENVIRONMENTS")
        if env_var:
            try:
                envs = json.loads(env_var)
                logger.info(f"[TaskGenerator] Using SAMPLING_ENVIRONMENTS from env var: {envs}")
                return envs
            except json.JSONDecodeError:
                logger.warning(f"[TaskGenerator] Invalid SAMPLING_ENVIRONMENTS env var, using SystemConfig")
        
        # Priority 2: SystemConfig database
        try:
            url = f"{self.api_base_url}/api/v1/config/sampling-environments"
            session = await self._get_session()
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    envs = response.get("environments", [])
                    logger.info(f"[TaskGenerator] Using SAMPLING_ENVIRONMENTS from SystemConfig: {envs}")
                    return envs
        except Exception as e:
            logger.error(f"[TaskGenerator] Error getting sampling environments: {e}")
        
        return []
    
    async def _get_env_task_ranges(self) -> Dict[str, Dict[str, any]]:
        """Get task ranges per environment with priority: env vars > SystemConfig.
        
        Returns:
            Dict mapping env names to range configs
        """
        # Priority 1: Environment variable
        env_var = os.getenv("ENV_TASK_RANGES")
        if env_var:
            try:
                ranges = json.loads(env_var)
                logger.info(f"[TaskGenerator] Using ENV_TASK_RANGES from env var")
                return ranges
            except json.JSONDecodeError:
                logger.warning(f"[TaskGenerator] Invalid ENV_TASK_RANGES env var, using SystemConfig")
        
        # Priority 2: SystemConfig database
        try:
            url = f"{self.api_base_url}/api/v1/config/env-task-ranges"
            session = await self._get_session()
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    ranges = response.get("ranges", {})
                    logger.info(f"[TaskGenerator] Using ENV_TASK_RANGES from SystemConfig")
                    return ranges
        except Exception as e:
            logger.error(f"[TaskGenerator] Error getting env task ranges: {e}")
        
        return {}
    
    async def _get_sampling_range(self, env: str) -> Tuple[int, int]:
        """Get sampling range for a specific environment.
        
        Args:
            env: Environment name
        
        Returns:
            Tuple of (start_id, end_id) for sampling
        """
        ranges = await self._get_env_task_ranges()
        env_config = ranges.get(env, {})
        sampling_range = env_config.get('sampling_range', [0, 0])
        
        if len(sampling_range) == 2:
            return tuple(sampling_range)
        else:
            logger.warning(f"[TaskGenerator] Invalid sampling_range for {env}: {sampling_range}")
            return (0, 0)
    
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
            session = await self._get_session()
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    response = await resp.json()
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
            session = await self._get_session()
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    response = await resp.json()
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
        """Generate tasks for missing task_ids in the sampling range.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            List of task dictionaries for missing task_ids
        """
        # Get sampling range for this environment
        start_id, end_id = await self._get_sampling_range(env)
        if start_id == end_id == 0:
            logger.warning(f"[TaskGenerator] No sampling range configured for {env}")
            return []
        
        # Get already sampled and queued task_ids
        sampled_ids = await self._get_sampled_task_ids(miner, env)
        queued_ids = await self._get_queued_task_ids(miner, env)
        
        # Calculate full range
        full_range = set(range(start_id, end_id))
        
        # Find missing task_ids
        existing_ids = sampled_ids | queued_ids
        missing_ids = full_range - existing_ids
        
        logger.info(
            f"[TaskGenerator] U{miner.uid} {env}: "
            f"range=[{start_id},{end_id}), sampled={len(sampled_ids)}, "
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
            }
            tasks.append(task)
        
        return tasks
    
    async def generate_tasks_for_miner(self, miner: MinerInfo) -> List[Dict]:
        """Generate all missing tasks for a miner across sampling environments.
        
        Args:
            miner: Miner information
        
        Returns:
            List of task dictionaries (empty list if all complete)
        """
        all_tasks = []
        
        # Get sampling environments (with priority: env var > SystemConfig)
        envs = await self._get_sampling_environments()
        if not envs:
            logger.warning(f"[TaskGenerator] No sampling environments configured")
            return []
        
        for env in envs:
            tasks = await self.generate_missing_tasks(miner, env)
            all_tasks.extend(tasks)
        
        logger.info(
            f"[TaskGenerator] Generated {len(all_tasks)} tasks for U{miner.uid} "
            f"across {len(envs)} environments"
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
            session = await self._get_session()
            
            async with session.post(url, json={"tasks": tasks}) as resp:
                if resp.status == 200:
                    response = await resp.json()
                    count = response.get("added_count", 0)
                    logger.info(f"[TaskGenerator] Submitted {count} tasks to pool")
                    return count
                else:
                    logger.error("[TaskGenerator] Failed to submit tasks to pool")
                    return 0
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error submitting tasks: {e}")
            return 0