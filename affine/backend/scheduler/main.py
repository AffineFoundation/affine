"""
Scheduler Service - Main Entry Point

Uses global MinersCacheManager to discover miners and generate batch sampling tasks.
"""

import asyncio
import traceback
from typing import Dict, List, Optional
from affine.backend.config import get_config
from affine.api.services.miners_cache import MinersCacheManager, MinerInfo
from affine.backend.scheduler.task_generator import TaskGenerator
from affine.core.setup import logger


class SchedulerService:
    """Main scheduler service: periodically generates sampling tasks
    
    Uses global MinersCacheManager to get miners and calls TaskGenerator to generate tasks.
    """
    
    def __init__(self, api_base_url: Optional[str] = None):
        """Initialize scheduler service
        
        Args:
            api_base_url: API server URL (default: from env or http://localhost:8000)
        """
        self.config = get_config(api_base_url)
        self.api_base_url = api_base_url or self.config.api_base_url
        
        # Use global MinersCacheManager
        self.miners_cache = MinersCacheManager.get_instance()
        self.task_generator = TaskGenerator(api_base_url=self.api_base_url)
        
        self.running = False
        self.task_generation_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_tasks_generated = 0
        self.total_tasks_submitted = 0
        self.total_api_errors = 0
    
    async def start(self):
        """Start scheduler service"""
        logger.info("[SchedulerService] Starting scheduler service...")
        
        self.running = True
        
        # Start background tasks
        self.task_generation_task = asyncio.create_task(self._task_generation_loop())
        self.metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("[SchedulerService] Scheduler service started")
    
    async def stop(self):
        """Stop scheduler service"""
        logger.info("[SchedulerService] Stopping scheduler service...")
        
        self.running = False
        
        # Cancel background tasks
        for task in [self.task_generation_task, self.metrics_task]:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self.task_generation_task,
            self.metrics_task,
            return_exceptions=True
        )
        
        logger.info("[SchedulerService] Scheduler service stopped")
    
    async def _get_all_miners(self) -> List[MinerInfo]:
        """Get all valid miners from global cache
        
        Returns:
            List of MinerInfo
        """
        miners_dict = await self.miners_cache.get_valid_miners()
        return list(miners_dict.values())
    
    async def _task_generation_loop(self):
        """Background task generation loop"""
        while self.running:
            try:
                # Get check interval config
                check_interval = await self.config.get("scheduler.check_interval")
                if check_interval is None:
                    check_interval = 600  # Default 10 minutes
                
                await asyncio.sleep(check_interval)
                
                # Generate tasks for all miners
                await self._generate_all_tasks()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SchedulerService] Error in task generation loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)  # Wait before retry
    
    async def _generate_all_tasks(self):
        """Generate and submit tasks for all miners"""
        try:
            # Get valid miners from global cache
            miners = await self._get_all_miners()
            
            if not miners:
                logger.warning("[SchedulerService] No valid miners found")
                return
            
            logger.info(f"[SchedulerService] Generating tasks for {len(miners)} miners...")
            
            for miner in miners:
                try:
                    # Generate missing tasks for this miner
                    tasks = await self.task_generator.generate_tasks_for_miner(miner)
                    
                    if not tasks:
                        continue
                    
                    # Submit tasks in batch
                    submitted = await self.task_generator.submit_tasks_to_pool(tasks)
                    self.total_tasks_generated += len(tasks)
                    self.total_tasks_submitted += submitted
                    
                    logger.info(
                        f"[SchedulerService] U{miner.uid}: generated={len(tasks)}, submitted={submitted}"
                    )
                
                except Exception as e:
                    logger.error(f"[SchedulerService] Error generating tasks for U{miner.uid}: {e}")
                    traceback.print_exc()
            
            logger.info(
                f"[SchedulerService] Task generation complete: "
                f"generated={self.total_tasks_generated}, submitted={self.total_tasks_submitted}"
            )
        
        except Exception as e:
            logger.error(f"[SchedulerService] Error in task generation: {e}")
            traceback.print_exc()
    
    async def _metrics_loop(self):
        """Background metrics logging loop"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                miner_count = self.miners_cache.get_miner_count()
                cache_age = self.miners_cache.get_cache_age()
                
                logger.info(
                    f"[SchedulerService] Metrics: "
                    f"miners={miner_count} "
                    f"cache_age={cache_age}s "
                    f"tasks_generated={self.total_tasks_generated} "
                    f"tasks_submitted={self.total_tasks_submitted} "
                    f"api_errors={self.total_api_errors}"
                )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SchedulerService] Error in metrics loop: {e}")
    
    def get_metrics(self) -> Dict:
        """Get current service metrics
        
        Returns:
            Metrics dictionary
        """
        return {
            "service": "scheduler",
            "running": self.running,
            "known_miners": self.miners_cache.get_miner_count(),
            "cache_age_seconds": self.miners_cache.get_cache_age(),
            "total_tasks_generated": self.total_tasks_generated,
            "total_tasks_submitted": self.total_tasks_submitted,
            "total_api_errors": self.total_api_errors,
        }


async def main():
    """Scheduler service main entry point"""
    import os
    
    # Get API base URL from environment
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    logger.info(f"[SchedulerService] Connecting to API: {api_base_url}")
    
    # Create and start service
    service = SchedulerService(api_base_url)
    
    try:
        await service.start()
        
        # Run initial task generation immediately
        logger.info("[SchedulerService] Running initial task generation...")
        await service._generate_all_tasks()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("[SchedulerService] Received interrupt signal")
    
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())