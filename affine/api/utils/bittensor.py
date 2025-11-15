"""
Bittensor Integration Utilities

Provides helper functions for querying bittensor metagraph metadata.
Miners should query metagraph directly instead of caching in database.
"""

import asyncio
from typing import Optional, Dict, Any
from functools import lru_cache


class BittensorClient:
    """Client for querying bittensor metagraph.
    
    This is a placeholder implementation. In production, replace with actual
    bittensor SDK calls to query the metagraph.
    """
    
    def __init__(self, netuid: int = 1):
        """Initialize bittensor client.
        
        Args:
            netuid: Network UID for the subnet
        """
        self.netuid = netuid
        self._metagraph_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 60  # Cache for 60 seconds
    
    async def get_metagraph(self) -> Dict[int, Dict[str, Any]]:
        """Get current metagraph.
        
        Returns a mapping of UID -> miner metadata.
        
        In production, this should call:
        ```python
        import bittensor as bt
        subtensor = bt.subtensor()
        metagraph = await subtensor.metagraph(self.netuid)
        ```
        
        Returns:
            Dict mapping uid to {hotkey, coldkey, stake, etc}
        """
        import time
        current_time = time.time()
        
        # Use cache if fresh
        if self._metagraph_cache and (current_time - self._cache_timestamp) < self._cache_ttl:
            return self._metagraph_cache
        
        # TODO: Replace with actual bittensor SDK call
        # For now, return empty dict as placeholder
        # In production:
        # import bittensor as bt
        # subtensor = bt.subtensor()
        # metagraph = subtensor.metagraph(self.netuid)
        # self._metagraph_cache = {
        #     uid: {
        #         'hotkey': metagraph.hotkeys[uid],
        #         'coldkey': metagraph.coldkeys[uid],
        #         'stake': metagraph.S[uid],
        #         'trust': metagraph.T[uid],
        #         'consensus': metagraph.C[uid],
        #         'incentive': metagraph.I[uid],
        #         'dividends': metagraph.D[uid],
        #         'emission': metagraph.E[uid],
        #     }
        #     for uid in range(len(metagraph.hotkeys))
        # }
        
        self._metagraph_cache = {}
        self._cache_timestamp = current_time
        return self._metagraph_cache
    
    async def get_hotkey_by_uid(self, uid: int) -> Optional[str]:
        """Get hotkey for a given UID.
        
        Args:
            uid: Miner UID
            
        Returns:
            Hotkey string or None if UID not found
        """
        metagraph = await self.get_metagraph()
        miner_info = metagraph.get(uid)
        return miner_info['hotkey'] if miner_info else None
    
    async def get_uid_by_hotkey(self, hotkey: str) -> Optional[int]:
        """Get UID for a given hotkey.
        
        Args:
            hotkey: Miner hotkey
            
        Returns:
            UID or None if hotkey not found
        """
        metagraph = await self.get_metagraph()
        for uid, info in metagraph.items():
            if info['hotkey'] == hotkey:
                return uid
        return None
    
    async def get_miner_info(self, uid: int) -> Optional[Dict[str, Any]]:
        """Get full miner information by UID.
        
        Args:
            uid: Miner UID
            
        Returns:
            Dict with miner metadata or None if not found
        """
        metagraph = await self.get_metagraph()
        return metagraph.get(uid)


# Global client instance
_bittensor_client: Optional[BittensorClient] = None


def get_bittensor_client(netuid: int = 1) -> BittensorClient:
    """Get or create bittensor client singleton.
    
    Args:
        netuid: Network UID (default: 1)
        
    Returns:
        BittensorClient instance
    """
    global _bittensor_client
    if _bittensor_client is None:
        _bittensor_client = BittensorClient(netuid=netuid)
    return _bittensor_client


async def query_hotkey_by_uid(uid: int, netuid: int = 1) -> Optional[str]:
    """Query hotkey for a given UID from bittensor metagraph.
    
    This is the main function to use in API endpoints for UID -> hotkey mapping.
    
    Args:
        uid: Miner UID
        netuid: Network UID (default: 1)
        
    Returns:
        Hotkey string or None if UID not found
    """
    client = get_bittensor_client(netuid)
    return await client.get_hotkey_by_uid(uid)


async def query_uid_by_hotkey(hotkey: str, netuid: int = 1) -> Optional[int]:
    """Query UID for a given hotkey from bittensor metagraph.
    
    Args:
        hotkey: Miner hotkey
        netuid: Network UID (default: 1)
        
    Returns:
        UID or None if hotkey not found
    """
    client = get_bittensor_client(netuid)
    return await client.get_uid_by_hotkey(hotkey)


async def query_miner_metadata(uid: int, netuid: int = 1) -> Optional[Dict[str, Any]]:
    """Query full miner metadata from bittensor metagraph.
    
    Args:
        uid: Miner UID
        netuid: Network UID (default: 1)
        
    Returns:
        Dict with miner metadata or None if not found
    """
    client = get_bittensor_client(netuid)
    return await client.get_miner_info(uid)