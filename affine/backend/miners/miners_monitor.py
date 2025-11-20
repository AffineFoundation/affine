"""
Miners Monitor Service

Monitors and validates miners with anti-plagiarism detection.
Persists validation state to Miners table.
"""

import os
import json
import time
import asyncio
import aiohttp
import logging
from typing import Dict, Optional, Set
from dataclasses import dataclass
from huggingface_hub import HfApi

from affine.utils.subtensor import get_subtensor
from affine.core.setup import NETUID
from affine.database.dao.miners import MinersDAO


logger = logging.getLogger(__name__)


@dataclass
class MinerInfo:
    """Miner information data class"""
    uid: int
    hotkey: str
    model: str
    revision: str
    chute_id: str
    chute_slug: str = ""
    block: int = 0
    is_valid: bool = False
    invalid_reason: Optional[str] = None
    
    def key(self) -> str:
        """Generate unique key: hotkey#revision"""
        return f"{self.hotkey}#{self.revision}"


class MinersMonitor:
    """Miners monitor and validation service
    
    Responsibilities:
    1. Discover miners from metagraph
    2. Validate chute status, revision, and model weights
    3. Detect plagiarism via model hash comparison
    4. Persist validation results to database
    """
    
    _instance: Optional['MinersMonitor'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, refresh_interval_seconds: int = 300):
        """Initialize monitor
        
        Args:
            refresh_interval_seconds: Auto-refresh interval in seconds
        """
        self.dao = MinersDAO()
        self.refresh_interval_seconds = refresh_interval_seconds
        self.last_update: int = 0
        
        # Caches
        self.weights_cache: Dict[tuple, tuple] = {}  # (model, revision) -> (sha_set, timestamp)
        self.weights_ttl = 3600  # 1 hour
        
        # Background task management
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        
        logger.info("[MinersMonitor] Initialized")
    
    @classmethod
    def get_instance(cls) -> 'MinersMonitor':
        """Get global singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    async def initialize(cls, refresh_interval_seconds: int = 300) -> 'MinersMonitor':
        """Initialize global singleton and start background tasks"""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(refresh_interval_seconds=refresh_interval_seconds)
                await cls._instance.refresh_miners()
                await cls._instance.start_background_tasks()
        return cls._instance
    
    async def _refresh_loop(self):
        """Background refresh loop"""
        while self._running:
            try:
                await self.refresh_miners()
                await asyncio.sleep(self.refresh_interval_seconds)
            except Exception as e:
                logger.error(f"[MinersMonitor] Error in refresh loop: {e}", exc_info=True)
    
    async def start_background_tasks(self):
        """Start background refresh tasks"""
        if self._running:
            logger.warning("[MinersMonitor] Background tasks already running")
            return
        
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info(f"[MinersMonitor] Background refresh started (interval={self.refresh_interval_seconds}s)")
    
    async def stop_background_tasks(self):
        """Stop background refresh tasks"""
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("[MinersMonitor] Background tasks stopped")
    
    def _load_blacklist(self) -> set:
        """Load blacklisted hotkeys from environment"""
        blacklist_str = os.getenv("AFFINE_MINER_BLACKLIST", "").strip()
        if not blacklist_str:
            return set()
        return {hk.strip() for hk in blacklist_str.split(",") if hk.strip()}
    
    async def _get_weights_shas(self, model_id: str, revision: Optional[str] = None) -> Optional[Set[str]]:
        """Get model weights SHA256 hashes from HuggingFace
        
        Args:
            model_id: HuggingFace model repo
            revision: Git commit hash
            
        Returns:
            Set of SHA256 hashes or None if failed
        """
        key = (model_id, revision)
        now = time.time()
        cached = self.weights_cache.get(key)
        
        if cached and now - cached[1] < self.weights_ttl:
            return cached[0]
        
        try:
            def _repo_info():
                return HfApi(token=os.getenv("HF_TOKEN")).repo_info(
                    repo_id=model_id,
                    repo_type="model",
                    revision=revision,
                    files_metadata=True,
                )
            
            info = await asyncio.to_thread(_repo_info)
            siblings = getattr(info, "siblings", None) or []
            
            def _name(s):
                return getattr(s, "rfilename", None) or getattr(s, "path", "") or ""
            
            shas = {
                str(getattr(s, "lfs", {})["sha256"])
                for s in siblings
                if (
                    isinstance(getattr(s, "lfs", None), dict)
                    and _name(s) is not None
                    and _name(s).endswith(".safetensors")
                    and "sha256" in getattr(s, "lfs", {})
                )
            }
            
            # Compute total hash (concatenate all SHAs and hash again)
            if shas:
                import hashlib
                total_hash = hashlib.sha256("".join(sorted(shas)).encode()).hexdigest()
                self.weights_cache[key] = (total_hash, now)
                return total_hash
            
            self.weights_cache[key] = (None, now)
            return None
            
        except Exception as e:
            logger.debug(f"Failed to fetch weights for {model_id}@{revision}: {e}")
            self.weights_cache[key] = (None, now)
            return None
    
    async def _get_chute(self, chute_id: str) -> Optional[Dict]:
        """Get chute info from Chutes API
        
        Args:
            chute_id: Chute deployment ID
            
        Returns:
            Chute info dict or None if failed
        """
        url = f"https://api.chutes.ai/chutes/{chute_id}"
        token = os.getenv("CHUTES_API_KEY", "")
        
        if not token:
            logger.warning("CHUTES_API_KEY not configured")
            return None
        
        headers = {"Authorization": token}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return None
                    
                    info = await resp.json()
                    # Remove unnecessary fields
                    for k in ("readme", "cords", "tagline", "instances"):
                        info.pop(k, None)
                    info.get("image", {}).pop("readme", None)
                    
                    return info
        except Exception as e:
            logger.debug(f"Failed to fetch chute {chute_id}: {e}")
            return None
    
    async def _validate_miner(
        self,
        uid: int,
        hotkey: str,
        model: str,
        revision: str,
        chute_id: str,
        block: int,
    ) -> MinerInfo:
        """Validate a single miner
        
        Args:
            uid: Miner UID
            hotkey: Miner hotkey
            model: Model repo
            revision: Git commit hash
            chute_id: Chute deployment ID
            block: Block when miner committed
            current_block: Current blockchain block
            
        Returns:
            MinerInfo with validation result
        """
        info = MinerInfo(
            uid=uid,
            hotkey=hotkey,
            model=model,
            revision=revision,
            chute_id=chute_id,
            block=block,
        )
        
        # Fetch chute info
        chute = await self._get_chute(chute_id)
        if not chute:
            info.is_valid = False
            info.invalid_reason = "chute_fetch_failed"
            return info
        
        # Check chute status
        chute_status = "hot" if chute.get("hot", False) else "cold"
        if chute_status != "hot":
            info.is_valid = False
            info.invalid_reason = "chute_not_hot"
            info.chute_slug = chute.get("slug", "")
            return info
        
        # Check model name matches
        chute_name = chute.get("name")
        if model != chute_name:
            info.is_valid = False
            info.invalid_reason = f"model_mismatch:chute={chute_name}"
            info.chute_slug = chute.get("slug", "")
            return info
        
        # Check revision matches
        chute_revision = chute.get("revision")
        if chute_revision and revision != chute_revision:
            info.is_valid = False
            info.invalid_reason = f"revision_mismatch:chute={chute_revision}"
            info.chute_slug = chute.get("slug", "")
            return info
        
        # All checks passed
        info.is_valid = True
        info.chute_slug = chute.get("slug", "")
        
        return info
    
    async def _detect_plagiarism(self, miners: list[MinerInfo]) -> list[MinerInfo]:
        """Detect plagiarism by checking duplicate model hashes
        
        Args:
            miners: List of validated miners
            
        Returns:
            Updated miners list with plagiarism detection
        """
        # Fetch model hashes concurrently
        tasks = []
        for miner in miners:
            if miner.is_valid:
                tasks.append(self._get_weights_shas(miner.model, miner.revision))
            else:
                tasks.append(asyncio.sleep(0, result=None))
        
        hashes = await asyncio.gather(*tasks)
        
        # Map hash -> list of (block, uid, miner)
        hash_to_miners: Dict[str, list] = {}
        for miner, model_hash in zip(miners, hashes):
            if model_hash:
                if model_hash not in hash_to_miners:
                    hash_to_miners[model_hash] = []
                hash_to_miners[model_hash].append((miner.block, miner.uid, miner))
        
        # For each hash group, keep only the earliest miner
        for model_hash, group in hash_to_miners.items():
            if len(group) <= 1:
                continue
            
            # Sort by block (earliest first)
            group.sort(key=lambda x: x[0])
            earliest_block, earliest_uid, _ = group[0]
            
            # Mark others as invalid
            for block, uid, miner in group[1:]:
                if miner.is_valid:
                    miner.is_valid = False
                    miner.invalid_reason = f"model_hash_duplicate:earliest_uid={earliest_uid}"
                    logger.info(
                        f"Detected plagiarism: uid={uid} copied from uid={earliest_uid} "
                        f"(hash={model_hash[:16]}...)"
                    )
        
        return miners
    
    async def refresh_miners(self) -> Dict[str, MinerInfo]:
        """Refresh and validate all miners
        
        Returns:
            Dict of valid miners {key: MinerInfo}
        """
        try:
            logger.info("[MinersMonitor] Refreshing miners from metagraph...")
            
            # Get metagraph and commits
            subtensor = await get_subtensor()
            meta = await subtensor.metagraph(NETUID)
            commits = await subtensor.get_all_revealed_commitments(NETUID)
            
            current_block = await subtensor.get_current_block()
            
            # Load blacklist
            blacklist = self._load_blacklist()
            
            # Discover and validate miners
            miners = []
            for uid in range(len(meta.hotkeys)):
                hotkey = meta.hotkeys[uid]
                
                # Check blacklist
                if hotkey in blacklist:
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason="blacklisted"
                    ))
                    continue
                
                # Check for commit
                if hotkey not in commits:
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason="no_commit"
                    ))
                    continue
                
                try:
                    block, commit_data = commits[hotkey][-1]
                    data = json.loads(commit_data)
                    
                    model = data.get("model", "")
                    revision = data.get("revision", "")
                    chute_id = data.get("chute_id", "")
                    
                    # Check if all required fields present
                    if not model or not revision or not chute_id:
                        miners.append(MinerInfo(
                            uid=uid,
                            hotkey=hotkey,
                            model=model,
                            revision=revision,
                            chute_id=chute_id,
                            block=int(block) if uid != 0 else 0,
                            is_valid=False,
                            invalid_reason="incomplete_commit:missing_fields"
                        ))
                        continue
                    
                    # Validate miner
                    miner_info = await self._validate_miner(
                        uid=uid,
                        hotkey=hotkey,
                        model=model,
                        revision=revision,
                        chute_id=chute_id,
                        block=int(block) if uid != 0 else 0,
                    )
                    
                    miners.append(miner_info)
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON in commit for uid={uid}: {e}")
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason="invalid_json_commit"
                    ))
                except Exception as e:
                    logger.debug(f"Failed to validate uid={uid}: {e}")
                    miners.append(MinerInfo(
                        uid=uid,
                        hotkey=hotkey,
                        model="",
                        revision="",
                        chute_id="",
                        block=0,
                        is_valid=False,
                        invalid_reason=f"validation_error:{str(e)[:50]}"
                    ))
            
            # Detect plagiarism
            miners = await self._detect_plagiarism(miners)
            
            # Persist to database
            for miner in miners:
                model_hash = await self._get_weights_shas(miner.model, miner.revision) or ""
                
                await self.dao.save_miner(
                    uid=miner.uid,
                    hotkey=miner.hotkey,
                    model=miner.model,
                    revision=miner.revision,
                    chute_id=miner.chute_id,
                    chute_slug=miner.chute_slug,
                    model_hash=model_hash,
                    chute_status="hot" if miner.is_valid else "cold",
                    is_valid=miner.is_valid,
                    invalid_reason=miner.invalid_reason,
                    block_number=current_block,
                    first_block=miner.block,
                )
            
            valid_miners = {m.key(): m for m in miners if m.is_valid}
            
            self.last_update = int(time.time())
            
            logger.info(
                f"[MinersMonitor] Refreshed {len(miners)} miners "
                f"({len(valid_miners)} valid, {len(miners) - len(valid_miners)} invalid)"
            )
            
            return valid_miners
            
        except Exception as e:
            logger.error(f"[MinersMonitor] Failed to refresh miners: {e}", exc_info=True)
            return {}
    
    async def get_valid_miners(self, force_refresh: bool = False) -> Dict[str, MinerInfo]:
        """Get current valid miner list
        
        Args:
            force_refresh: Whether to force refresh
            
        Returns:
            Miners dictionary {key: MinerInfo}
        """
        # Query from database
        miners_data = await self.dao.get_valid_miners()
        
        # Convert to MinerInfo
        result = {}
        for data in miners_data:
            info = MinerInfo(
                uid=data['uid'],
                hotkey=data['hotkey'],
                model=data['model'],
                revision=data['revision'],
                chute_id=data['chute_id'],
                chute_slug=data.get('chute_slug', ''),
                block=data.get('first_block', 0),
                is_valid=True,
            )
            result[info.key()] = info
        
        return result