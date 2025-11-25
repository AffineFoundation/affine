import os
import json
import asyncio
from typing import Dict, List, Optional, Union
from affine.core.models import Miner
from affine.core.setup import NETUID
from affine.utils.subtensor import get_subtensor
from affine.utils.api_client import get_chute_info

logger = __import__("logging").getLogger("affine")


async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: object = None,
) -> Dict[int, "Miner"]:
    """Query miner information from blockchain.
    
    Simplified version for miner SDK usage - returns basic miner info from blockchain commits.
    For validator use cases with filtering logic, refer to affine.src.monitor.miners_monitor.
    
    Args:
        uids: Miner UID(s) to query. If None, queries all UIDs.
        netuid: Network UID (default: from NETUID config)
        meta: Optional metagraph object (will be fetched if not provided)
        
    Returns:
        Dict mapping UID to Miner info. Only includes miners with valid commits.
        
    Example:
        >>> miner = await af.miners(7)
        >>> if miner:
        >>>     print(miner[7].model)
    """
    sub = await get_subtensor()
    meta = meta or await sub.metagraph(netuid)
    commits = await sub.get_all_revealed_commitments(netuid)
    
    if uids is None:
        uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids = [uids]
    
    meta_sem = asyncio.Semaphore(int(os.getenv("AFFINE_META_CONCURRENCY", "12")))

    async def _fetch_miner(uid: int) -> Optional["Miner"]:
        try:
            hotkey = meta.hotkeys[uid]
            if hotkey not in commits:
                return None

            block, commit_data = commits[hotkey][-1]
            block = 0 if uid == 0 else block
            data = json.loads(commit_data)
            
            model = data.get("model")
            miner_revision = data.get("revision")
            chute_id = data.get("chute_id")

            if not model or not miner_revision or not chute_id:
                return None


            async with meta_sem:
                chute = await get_chute_info(chute_id)

            if not chute or not chute.get("hot", False):
                return None

            return Miner(
                uid=uid,
                hotkey=hotkey,
                model=model,
                block=int(block),
                revision=miner_revision,
                slug=chute.get("slug"),
                chute=chute,
            )
        except Exception as e:
            logger.trace(f"Failed to fetch miner uid={uid}: {e}")
            return None

    results = await asyncio.gather(*(_fetch_miner(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}

    return output
