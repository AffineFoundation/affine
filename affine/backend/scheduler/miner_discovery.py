"""
Miner Discovery Module

Discovers miners from Bittensor metagraph and manages miner state.
"""

import asyncio
from typing import Dict, Set, Optional
from dataclasses import dataclass
from affine.core.miners import miners as get_miners
from affine.utils.subtensor import get_subtensor
from affine.core.setup import NETUID, logger


@dataclass
class MinerInfo:
    """Information about a discovered miner."""
    uid: int
    hotkey: str
    model: str
    revision: str
    slug: Optional[str]
    block: int
    
    def key(self) -> str:
        """Get unique key for this miner configuration."""
        return f"{self.hotkey}#{self.model}#{self.revision}"


class MinerDiscovery:
    """Discovers and tracks miners from Bittensor network."""
    
    def __init__(self):
        self.known_miners: Dict[str, MinerInfo] = {}  # key -> MinerInfo
        self.uid_to_key: Dict[int, str] = {}  # uid -> key
    
    async def discover_miners(self) -> Dict[str, MinerInfo]:
        """Discover current miners from metagraph.
        
        Returns:
            Dict mapping miner key to MinerInfo
        """
        try:
            # Get current metagraph
            subtensor = await get_subtensor()
            meta = await subtensor.metagraph(NETUID)
            
            # Get queryable miners (hot chutes, valid, not gated)
            miners_map = await get_miners(meta=meta, netuid=NETUID, check_validity=True)
            
            discovered = {}
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
                discovered[key] = info
                self.uid_to_key[uid] = key
            
            logger.info(f"[MinerDiscovery] Discovered {len(discovered)} queryable miners")
            return discovered
        
        except Exception as e:
            logger.error(f"[MinerDiscovery] Failed to discover miners: {e}")
            return {}
    
    def get_changes(self, new_miners: Dict[str, MinerInfo]) -> tuple[Set[str], Set[str], Set[str]]:
        """Compare new miners with known miners to find changes.
        
        Args:
            new_miners: Newly discovered miners
        
        Returns:
            Tuple of (added_keys, removed_keys, changed_keys)
        """
        old_keys = set(self.known_miners.keys())
        new_keys = set(new_miners.keys())
        
        added = new_keys - old_keys
        removed = old_keys - new_keys
        changed = set()
        
        # Check for UID changes (same key but different UID)
        for key in old_keys & new_keys:
            if self.known_miners[key].uid != new_miners[key].uid:
                changed.add(key)
        
        return added, removed, changed
    
    def update_known_miners(self, new_miners: Dict[str, MinerInfo]):
        """Update internal state with new miners.
        
        Args:
            new_miners: New miner information
        """
        self.known_miners = new_miners.copy()
        
        # Rebuild UID mapping
        self.uid_to_key.clear()
        for key, info in self.known_miners.items():
            self.uid_to_key[info.uid] = key
    
    def get_miner_by_uid(self, uid: int) -> Optional[MinerInfo]:
        """Get miner info by UID.
        
        Args:
            uid: Miner UID
        
        Returns:
            MinerInfo if found, None otherwise
        """
        key = self.uid_to_key.get(uid)
        if key:
            return self.known_miners.get(key)
        return None
    
    def get_miner_by_key(self, key: str) -> Optional[MinerInfo]:
        """Get miner info by key.
        
        Args:
            key: Miner key (hotkey#model#revision)
        
        Returns:
            MinerInfo if found, None otherwise
        """
        return self.known_miners.get(key)