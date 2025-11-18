"""
Miners Cache Manager

Global singleton service maintaining current valid miner list for reuse across components.
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from affine.core.miners import miners as get_miners
from affine.utils.subtensor import get_subtensor
from affine.core.setup import NETUID

logger = logging.getLogger(__name__)


@dataclass
class MinerInfo:
    """Miner information data class"""
    uid: int
    hotkey: str
    model: str
    revision: str
    slug: Optional[str] = None
    block: int = 0
    
    def key(self) -> str:
        """Generate unique key: hotkey#model#revision"""
        return f"{self.hotkey}#{self.model}#{self.revision}"


class MinersCacheManager:
    """Global miner list cache manager
    
    Provides unified miner discovery and caching service to avoid code duplication.
    
    Use cases:
    - TaskPoolManager: weighted random task selection
    - TaskGenerator: task generation and invalid task cleanup
    - Scheduler: miner discovery and change detection
    - API endpoints: query miner information
    """
    
    _instance: Optional['MinersCacheManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, refresh_interval_seconds: int = 300):
        """Initialize cache manager
        
        Args:
            refresh_interval_seconds: Auto-refresh interval in seconds
        """
        self.miners: Dict[str, MinerInfo] = {}  # key: hotkey#model#revision -> MinerInfo
        self.uid_to_key: Dict[int, str] = {}  # uid -> key
        self.last_update: int = 0
        self.update_interval: int = refresh_interval_seconds
        self.refresh_interval_seconds = refresh_interval_seconds
        self.lock = asyncio.Lock()  # Refresh lock
        
        # Background task management
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        
        logger.info("[MinersCacheManager] Initialized")
    
    @classmethod
    def get_instance(cls) -> 'MinersCacheManager':
        """Get global singleton instance
        
        Returns:
            MinersCacheManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    async def initialize(cls, refresh_interval_seconds: int = 300) -> 'MinersCacheManager':
        """Initialize global singleton and start background tasks
        
        Args:
            refresh_interval_seconds: Refresh interval in seconds
            
        Returns:
            MinersCacheManager instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(refresh_interval_seconds=refresh_interval_seconds)
                # Refresh immediately
                await cls._instance.refresh_miners()
                # Start background tasks
                await cls._instance.start_background_tasks()
        return cls._instance
    
    async def _refresh_loop(self):
        """Background refresh loop"""
        while self._running:
            try:
                await asyncio.sleep(self.refresh_interval_seconds)
                await self.refresh_miners()
            except Exception as e:
                logger.error(f"[MinersCacheManager] Error in refresh loop: {e}", exc_info=True)
    
    async def start_background_tasks(self):
        """Start background refresh tasks"""
        if self._running:
            logger.warning("[MinersCacheManager] Background tasks already running")
            return
        
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info(f"[MinersCacheManager] Background refresh started (interval={self.refresh_interval_seconds}s)")
    
    async def stop_background_tasks(self):
        """Stop background refresh tasks"""
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("[MinersCacheManager] Background tasks stopped")
    
    async def refresh_miners(self) -> Dict[str, MinerInfo]:
        """Refresh miner list from metagraph
        
        Returns:
            Updated miners dictionary
        """
        async with self.lock:
            try:
                logger.info("[MinersCacheManager] Refreshing miners from metagraph...")
                
                # Get metagraph
                subtensor = await get_subtensor()
                meta = await subtensor.metagraph(NETUID)
                
                # Get queryable miners (hot chutes, valid, not gated)
                miners_map = await get_miners(meta=meta, netuid=NETUID, check_validity=True)
                
                # Build new miners dictionary
                new_miners = {}
                new_uid_to_key = {}
                
                for uid, miner in miners_map.items():
                    info = MinerInfo(
                        uid=uid,
                        hotkey=miner.hotkey,
                        model=miner.model,
                        revision=miner.revision,
                        slug=miner.slug,
                        block=miner.block or 0,
                    )
                    
                    key = info.key()
                    new_miners[key] = info
                    new_uid_to_key[uid] = key
                
                # Update internal state
                self.miners = new_miners
                self.uid_to_key = new_uid_to_key
                self.last_update = int(time.time())
                
                logger.info(
                    f"[MinersCacheManager] Refreshed {len(self.miners)} miners "
                    f"(UIDs: {sorted(self.uid_to_key.keys())})"
                )
                
                return self.miners.copy()
            
            except Exception as e:
                logger.error(f"[MinersCacheManager] Failed to refresh miners: {e}")
                # Return old cache
                return self.miners.copy()
    
    async def get_valid_miners(self, force_refresh: bool = False) -> Dict[str, MinerInfo]:
        """Get current valid miner list (auto-refresh)
        
        Args:
            force_refresh: Whether to force refresh
        
        Returns:
            Miners dictionary {key: MinerInfo}
        """
        now = int(time.time())
        
        # Check if refresh needed
        if force_refresh or (now - self.last_update) > self.update_interval:
            await self.refresh_miners()
        
        return self.miners.copy()
    
    async def get_miner_info(self, hotkey: str, revision: str) -> Optional[MinerInfo]:
        """Get single miner information
        
        Args:
            hotkey: Miner hotkey
            revision: Model revision
        
        Returns:
            MinerInfo or None
        """
        # Ensure cache is up-to-date (allow 5 minute staleness)
        await self.get_valid_miners()
        
        key = f"{hotkey}#{revision}"
        return self.miners.get(key)
    
    async def get_miner_by_uid(self, uid: int) -> Optional[MinerInfo]:
        """Get miner information by UID
        
        Args:
            uid: Miner UID
        
        Returns:
            MinerInfo or None
        """
        # Ensure cache is up-to-date
        await self.get_valid_miners()
        
        key = self.uid_to_key.get(uid)
        if key:
            return self.miners.get(key)
        return None
    
    def get_miner_count(self) -> int:
        """Get current cached miner count
        
        Returns:
            Miner count
        """
        return len(self.miners)
    
    def get_cache_age(self) -> int:
        """Get cache age in seconds
        
        Returns:
            Seconds since last update
        """
        return int(time.time()) - self.last_update