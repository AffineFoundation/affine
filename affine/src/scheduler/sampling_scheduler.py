"""
Sampling Scheduler

Periodically rotates sampling lists and cleans up removed tasks from TaskPool.
"""

import time
import asyncio
import logging
from typing import List, Optional

from affine.core.setup import logger
from affine.core.sampling_list import SamplingListManager
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.task_pool import TaskPoolDAO
from affine.database.dao.miners import MinersDAO


class SamplingScheduler:
    """Sampling list rotation scheduler."""
    
    def __init__(
        self,
        system_config_dao: Optional[SystemConfigDAO] = None,
        task_pool_dao: Optional[TaskPoolDAO] = None,
        sampling_list_manager: Optional[SamplingListManager] = None
    ):
        self.config_dao = system_config_dao or SystemConfigDAO()
        self.task_pool_dao = task_pool_dao or TaskPoolDAO()
        self.manager = sampling_list_manager or SamplingListManager()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start rotation scheduler."""
        logger.info("Starting sampling list rotation scheduler")
        self._running = True
        self._task = asyncio.create_task(self._rotation_loop())
    
    async def stop(self):
        """Stop rotation scheduler."""
        logger.info("Stopping sampling list rotation scheduler")
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _rotation_loop(self):
        """Rotation loop - checks every hour."""
        while self._running:
            try:
                await self._check_and_rotate_all_envs()
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                logger.info("Rotation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Rotation loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _check_and_rotate_all_envs(self):
        """Check all environments and rotate if needed.
        
        IMPORTANT: Only rotates environments with enabled_for_scoring=true
        """
        environments = await self.config_dao.get_param_value('environments', {})
        current_time = int(time.time())
        
        for env_name, env_config in environments.items():
            try:
                # KEY CHANGE: Only rotate if enabled_for_scoring=true
                if not env_config.get('enabled_for_scoring'):
                    logger.info(f"Skipping rotation for {env_name}: scoring disabled")
                    continue
                
                sampling_config = env_config.get('sampling_config')
                if not sampling_config:
                    logger.info(f"Skipping rotation for {env_name}: no sampling_config")
                    continue
                
                # Check if rotation is enabled
                rotation_enabled = sampling_config.get('rotation_enabled', True)
                if not rotation_enabled:
                    logger.info(f"Skipping rotation for {env_name}: rotation disabled")
                    continue
                
                # Check if rotation is needed
                last_rotation = sampling_config.get('last_rotation_at', 0)
                rotation_interval = sampling_config.get('rotation_interval', 3600)
                rotation_count = sampling_config.get('rotation_count', 0)
                
                # Skip if rotation_count=0
                if rotation_count == 0:
                    logger.info(f"Skipping rotation for {env_name}: rotation_count=0")
                    continue
                
                # Check if interval elapsed
                if current_time - last_rotation >= rotation_interval:
                    await self._rotate_environment(env_name, sampling_config)
                    
            except Exception as e:
                logger.error(
                    f"Error checking rotation for {env_name}: {e}",
                    exc_info=True
                )
    
    async def _rotate_environment(self, env: str, sampling_config: dict):
        """Rotate sampling list for a single environment."""
        logger.info(f"Rotating sampling list for {env}")
        
        current_list = sampling_config['sampling_list']
        dataset_range = sampling_config['dataset_range']
        sampling_count = sampling_config['sampling_count']
        rotation_count = sampling_config['rotation_count']
        
        # 1. Execute rotation
        new_list, removed_ids, added_ids = await self.manager.rotate_sampling_list(
            env=env,
            current_list=current_list,
            dataset_range=dataset_range,
            sampling_count=sampling_count,
            rotation_count=rotation_count
        )
        
        logger.info(
            f"Rotated {env}: removed={len(removed_ids)}, added={len(added_ids)}, "
            f"new_size={len(new_list)}"
        )
        
        # 2. Update SystemConfig
        await self._update_sampling_config(env, new_list)
        
        # 3. Cleanup TaskPool for removed task IDs
        await self._cleanup_removed_tasks(env, removed_ids)
    
    async def _update_sampling_config(self, env: str, new_list: List[int]):
        """Update sampling_list in SystemConfig."""
        environments = await self.config_dao.get_param_value('environments', {})
        
        if env not in environments:
            logger.warning(f"Environment {env} not found in config during update")
            return
        
        environments[env]['sampling_config']['sampling_list'] = new_list
        environments[env]['sampling_config']['last_rotation_at'] = int(time.time())
        
        await self.config_dao.set_param(
            param_name='environments',
            param_value=environments,
            param_type='dict',
            description='Environment configurations with dynamic sampling',
            updated_by='sampling_scheduler'
        )
        
        logger.info(f"Updated sampling_list for {env} in SystemConfig")
    
    async def _cleanup_removed_tasks(self, env: str, removed_ids: List[int]):
        """Cleanup removed task IDs from TaskPool (pending only).
        
        Strategy: For each valid miner + removed task_id, delete pending tasks.
        No scan needed - use PK+SK direct deletion.
        """
        if not removed_ids:
            return
        
        # Get all valid miners
        miners_dao = MinersDAO()
        valid_miners = await miners_dao.get_valid_miners()
        
        # Delete pending tasks for each miner + task_id combination
        deleted_count = 0
        for miner in valid_miners:
            hotkey = miner['hotkey']
            revision = miner['revision']
            
            for task_id in removed_ids:
                # Construct PK and SK for pending task
                pk = self.task_pool_dao._make_pk(hotkey, revision)
                sk = self.task_pool_dao._make_sk(env, 'pending', task_id)
                
                # Try to delete (silent if not exists)
                try:
                    deleted = await self.task_pool_dao.delete(pk, sk)
                    if deleted:
                        deleted_count += 1
                except Exception as e:
                    logger.debug(
                        f"Failed to delete task {env}/{task_id} for miner "
                        f"{hotkey[:8]}...#{revision[:8]}...: {e}"
                    )
        
        logger.info(
            f"Cleaned up {deleted_count} pending tasks for {len(removed_ids)} "
            f"removed task IDs in {env}"
        )