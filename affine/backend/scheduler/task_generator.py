"""
Task Generator Module

Generates sequential sampling tasks for miners via API.
No local state - all task_id tracking is done via API endpoints.
"""

import time
from typing import Dict, Optional, List
from affine.backend.config import get_config
from affine.backend.scheduler.miner_discovery import MinerInfo
from affine.http_client import AsyncHTTPClient
from affine.setup import logger, get_env_names


class TaskGenerator:
    """Generates sequential sampling tasks for miners via API."""
    
    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize task generator.
        
        Args:
            api_base_url: API server URL (default: from config)
        """
        self.config = get_config(api_base_url)
        self.api_base_url = api_base_url or self.config.api_base_url
        self.http_client = AsyncHTTPClient(timeout=30)
        
        # Track last generation time per miner+env (in-memory only)
        # Format: {miner_key: {env: last_timestamp}}
        self.last_generated: Dict[str, Dict[str, float]] = {}
    
    async def _check_is_paused(self, miner_hotkey: str) -> bool:
        """Check if miner is paused via API.
        
        Args:
            miner_hotkey: Miner hotkey
        
        Returns:
            True if paused, False otherwise
        """
        try:
            url = f"{self.api_base_url}/api/v1/miners/{miner_hotkey}/is-paused"
            response = await self.http_client.get(url)
            
            if response:
                return response.get("is_paused", False)
            else:
                logger.warning(f"[TaskGenerator] Failed to check pause status for {miner_hotkey[:8]}...")
                return False
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error checking pause status: {e}")
            return False
    
    async def _get_completion_stats(self, miner: MinerInfo, env: str) -> Optional[Dict]:
        """Get completion statistics via API.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            Stats dict or None if error
        """
        try:
            url = f"{self.api_base_url}/api/v1/miners/{miner.hotkey}/stats"
            params = {
                "model_revision": miner.revision,
                "env": env
            }
            response = await self.http_client.get(url, params=params)
            
            if response:
                return response
            else:
                logger.warning(
                    f"[TaskGenerator] Failed to get stats for U{miner.uid} {env}"
                )
                return None
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error getting stats: {e}")
            return None
    
    async def _get_next_task_id(self, miner: MinerInfo, env: str) -> Optional[int]:
        """Get next task_id via API.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            Next task_id or None if error
        """
        try:
            url = f"{self.api_base_url}/api/v1/tasks/next-task-id"
            params = {
                "miner_hotkey": miner.hotkey,
                "model_revision": miner.revision,
                "env": env
            }
            response = await self.http_client.get(url, params=params)
            
            if response:
                return response.get("next_task_id")
            else:
                logger.warning(
                    f"[TaskGenerator] Failed to get next task_id for U{miner.uid} {env}"
                )
                return None
        
        except Exception as e:
            logger.error(f"[TaskGenerator] Error getting next task_id: {e}")
            return None
    
    async def should_generate_task(self, miner: MinerInfo, env: str) -> bool:
        """Check if a task should be generated for this miner+env.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            True if task should be generated
        """
        # Check if miner is paused
        is_paused = await self._check_is_paused(miner.hotkey)
        if is_paused:
            logger.debug(f"[TaskGenerator] U{miner.uid} is paused, skip task generation")
            return False
        
        # Get configuration
        env_key = f"env.{env.replace(':', '_')}"
        daily_rate = await self.config.get(f"{env_key}.daily_rate")
        if daily_rate is None:
            daily_rate = 200  # Default fallback
        
        # Check completion rate - if not completed one round, accelerate 3x
        stats = await self._get_completion_stats(miner, env)
        if stats:
            completion_pct = stats.get("completion_percentage", 0)
            if completion_pct < 100:
                daily_rate *= 3  # Accelerate 3x if not completed one round
                logger.debug(
                    f"[TaskGenerator] U{miner.uid} {env} at {completion_pct:.1f}%, "
                    f"accelerating to {daily_rate}/day"
                )
        
        # Calculate sampling interval (seconds between samples)
        interval = 86400 / daily_rate  # 24h / daily_rate
        
        # Get last generation time
        miner_key = miner.key()
        if miner_key not in self.last_generated:
            return True  # First time, generate immediately
        
        last_time = self.last_generated[miner_key].get(env)
        if last_time is None:
            return True  # First time for this environment
        
        # Check if enough time has passed
        elapsed = time.time() - last_time
        return elapsed >= interval
    
    async def generate_task(self, miner: MinerInfo, env: str) -> Optional[Dict]:
        """Generate next task for miner+env.
        
        Args:
            miner: Miner information
            env: Environment name
        
        Returns:
            Task dictionary ready for API submission, or None if error
        """
        miner_key = miner.key()
        
        # Get next task_id from API
        next_task_id = await self._get_next_task_id(miner, env)
        if next_task_id is None:
            logger.error(
                f"[TaskGenerator] Failed to get next task_id for U{miner.uid} {env}"
            )
            return None
        
        # Generate task
        task = {
            "miner_hotkey": miner.hotkey,
            "model_revision": miner.revision,
            "model": miner.model,
            "env": env,
            "task_id": next_task_id,
            "priority": 0,
        }
        
        # Update last generation time
        if miner_key not in self.last_generated:
            self.last_generated[miner_key] = {}
        self.last_generated[miner_key][env] = time.time()
        
        logger.debug(
            f"[TaskGenerator] Generated task for U{miner.uid} {env} "
            f"task_id={task['task_id']}"
        )
        
        return task
    
    async def reset_miner(self, miner_key: str):
        """Reset last generation time for a miner (when model/revision changes).
        
        Args:
            miner_key: Miner key to reset
        """
        if miner_key in self.last_generated:
            del self.last_generated[miner_key]
            logger.info(f"[TaskGenerator] Reset generation time for {miner_key}")
    
    async def generate_tasks_for_miner(self, miner: MinerInfo) -> List[Dict]:
        """Generate tasks for a miner across all environments.
        
        Args:
            miner: Miner information
        
        Returns:
            List of task dictionaries (empty list if all skipped or errors)
        """
        tasks = []
        envs = get_env_names()
        
        for env in envs:
            if await self.should_generate_task(miner, env):
                task = await self.generate_task(miner, env)
                if task:  # Only add if generation succeeded
                    tasks.append(task)
        
        return tasks
    
    def get_state_summary(self) -> Dict:
        """Get summary of current generation state.
        
        Returns:
            Summary statistics
        """
        return {
            "total_miners": len(self.last_generated),
            "total_env_states": sum(len(env_times) for env_times in self.last_generated.values()),
        }