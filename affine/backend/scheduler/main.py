"""
Scheduler Service - Main Entry Point

Discovers miners and generates sequential sampling tasks via API.
"""

import asyncio
import traceback
from typing import Dict, Set, Optional
from affine.backend.config import get_config
from affine.backend.scheduler.miner_discovery import MinerDiscovery, MinerInfo
from affine.backend.scheduler.task_generator import TaskGenerator
from affine.http_client import AsyncHTTPClient
from affine.setup import logger


class SchedulerService:
    """Main scheduler service that coordinates miner discovery and task generation."""
    
    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize scheduler service.
        
        Args:
            api_base_url: API server URL (default: from env or http://localhost:8000)
        """
        self.config = get_config(api_base_url)
        self.api_base_url = api_base_url or self.config.api_base_url
        self.http_client = AsyncHTTPClient(timeout=30)
        
        self.discovery = MinerDiscovery()
        self.task_generator = TaskGenerator(api_base_url=self.api_base_url)
        
        self.running = False
        self.miner_refresh_task: Optional[asyncio.Task] = None
        self.task_generation_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_tasks_generated = 0
        self.total_api_errors = 0
    
    async def start(self):
        """Start the scheduler service."""
        logger.info("[SchedulerService] Starting scheduler service...")
        
        self.running = True
        
        # Initial miner discovery
        await self._refresh_miners()
        
        # Start background tasks
        self.miner_refresh_task = asyncio.create_task(self._miner_refresh_loop())
        self.task_generation_task = asyncio.create_task(self._task_generation_loop())
        self.metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("[SchedulerService] Scheduler service started")
    
    async def stop(self):
        """Stop the scheduler service."""
        logger.info("[SchedulerService] Stopping scheduler service...")
        
        self.running = False
        
        # Cancel background tasks
        for task in [self.miner_refresh_task, self.task_generation_task, self.metrics_task]:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self.miner_refresh_task,
            self.task_generation_task,
            self.metrics_task,
            return_exceptions=True
        )
        
        logger.info("[SchedulerService] Scheduler service stopped")
    
    async def _refresh_miners(self):
        """Discover miners and handle changes."""
        try:
            # Discover current miners
            new_miners = await self.discovery.discover_miners()
            
            # Detect changes
            added, removed, changed = self.discovery.get_changes(new_miners)
            
            # Handle changes
            if added:
                logger.info(f"[SchedulerService] New miners: {len(added)}")
                for key in added:
                    miner = new_miners[key]
                    logger.info(f"  + U{miner.uid} {key}")
            
            if removed:
                logger.info(f"[SchedulerService] Removed miners: {len(removed)}")
                for key in removed:
                    logger.info(f"  - {key}")
                    # Reset task state for removed miners
                    await self.task_generator.reset_miner(key)
            
            if changed:
                logger.info(f"[SchedulerService] Changed miners: {len(changed)}")
                for key in changed:
                    logger.info(f"  ~ {key}")
                    # Reset task state for changed miners
                    await self.task_generator.reset_miner(key)
            
            # Update known miners
            self.discovery.update_known_miners(new_miners)
        
        except Exception as e:
            logger.error(f"[SchedulerService] Error refreshing miners: {e}")
            traceback.print_exc()
    
    async def _miner_refresh_loop(self):
        """Background loop for refreshing miner list."""
        while self.running:
            try:
                # Get refresh interval from config
                interval = await self.config.get("scheduler.miner_refresh_interval")
                if interval is None:
                    interval = 1800  # Default 30 minutes
                
                await asyncio.sleep(interval)
                await self._refresh_miners()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SchedulerService] Error in miner refresh loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _task_generation_loop(self):
        """Background loop for generating tasks."""
        while self.running:
            try:
                # Get check interval from config
                check_interval = await self.config.get("scheduler.check_interval")
                if check_interval is None:
                    check_interval = 10  # Default 10 seconds
                
                await asyncio.sleep(check_interval)
                
                # Generate tasks for all known miners
                await self._generate_all_tasks()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SchedulerService] Error in task generation loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)  # Wait before retry
    
    async def _generate_all_tasks(self):
        """Generate tasks for all miners that need them."""
        for key, miner in self.discovery.known_miners.items():
            try:
                # Generate tasks for this miner
                tasks = await self.task_generator.generate_tasks_for_miner(miner)
                
                # Submit tasks to API
                for task in tasks:
                    success = await self._submit_task(task)
                    if success:
                        self.total_tasks_generated += 1
            
            except Exception as e:
                logger.error(f"[SchedulerService] Error generating tasks for {key}: {e}")
    
    async def _submit_task(self, task: Dict) -> bool:
        """Submit task to API.
        
        Args:
            task: Task dictionary
        
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.api_base_url}/api/v1/tasks"
            response = await self.http_client.post(url, json=task)
            
            if response:
                logger.debug(
                    f"[SchedulerService] Task submitted: "
                    f"{task['miner_hotkey'][:8]}... {task['env']} task_id={task['task_id']}"
                )
                return True
            else:
                logger.warning(f"[SchedulerService] Failed to submit task: {task}")
                self.total_api_errors += 1
                return False
        
        except Exception as e:
            logger.error(f"[SchedulerService] Error submitting task: {e}")
            self.total_api_errors += 1
            return False
    
    async def _metrics_loop(self):
        """Background loop for logging metrics."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                state_summary = self.task_generator.get_state_summary()
                
                logger.info(
                    f"[SchedulerService] Metrics: "
                    f"known_miners={len(self.discovery.known_miners)} "
                    f"tasks_generated={self.total_tasks_generated} "
                    f"api_errors={self.total_api_errors} "
                    f"task_states={state_summary['total_env_states']}"
                )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SchedulerService] Error in metrics loop: {e}")
    
    def get_metrics(self) -> Dict:
        """Get current service metrics.
        
        Returns:
            Dictionary of metrics
        """
        state_summary = self.task_generator.get_state_summary()
        
        return {
            "service": "scheduler",
            "running": self.running,
            "known_miners": len(self.discovery.known_miners),
            "total_tasks_generated": self.total_tasks_generated,
            "total_api_errors": self.total_api_errors,
            "task_states": state_summary,
        }


async def main():
    """Main entry point for scheduler service."""
    import os
    
    # Get API base URL from environment
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Create and start service
    service = SchedulerService(api_base_url)
    
    try:
        await service.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("[SchedulerService] Received interrupt signal")
    
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())