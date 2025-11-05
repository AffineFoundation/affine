"""Blacklist management for miners with suspicious behavior."""

import os
import time
import asyncio
import orjson
from typing import Dict, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

from aiobotocore.session import get_session
from botocore.config import Config

from affine.setup import logger

# R2 configuration (reuse from storage.py)
FOLDER = os.getenv("R2_FOLDER", "affine")
BUCKET = os.getenv("R2_BUCKET_ID", "00523074f51300584834607253cae0fa")
ACCESS = os.getenv("R2_WRITE_ACCESS_KEY_ID", "")
SECRET = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

BLACKLIST_KEY = "affine/verification/blacklist.json"
BLACKLIST_VERSION = "1.0.0"

# Local cache
CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR", Path.home() / ".cache" / "affine" / "verification"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BLACKLIST_CACHE = CACHE_DIR / "blacklist.json"


def get_client_ctx():
    """Get S3 client context for R2."""
    return get_session().create_client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS,
        aws_secret_access_key=SECRET,
        config=Config(max_pool_connections=256),
    )


@dataclass
class BlacklistEntry:
    """Blacklist entry for a miner."""

    hotkey: str
    model: str
    reason: str
    similarity_score: float
    block: int
    timestamp: float
    samples_tested: int
    details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert None to empty dict for JSON compatibility
        if d.get("details") is None:
            d["details"] = {}
        return d


class BlacklistManager:
    """Manage blacklist for miners."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._cache: Optional[Dict[str, BlacklistEntry]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl = 300  # 5 minutes

    async def load_blacklist(self, force_refresh: bool = False) -> Dict[str, BlacklistEntry]:
        """Load blacklist from R2 or cache.

        Args:
            force_refresh: Force refresh from R2 even if cache is valid

        Returns:
            Dictionary mapping hotkey to BlacklistEntry
        """
        async with self._lock:
            now = time.time()

            # Return cached data if still valid
            if not force_refresh and self._cache is not None and (now - self._cache_timestamp) < self._cache_ttl:
                logger.debug("Using cached blacklist")
                return self._cache

            # Try loading from R2
            try:
                async with get_client_ctx() as client:
                    response = await client.get_object(Bucket=FOLDER, Key=BLACKLIST_KEY)
                    body = await response["Body"].read()
                    data = orjson.loads(body)

                    # Validate version
                    version = data.get("version", "0.0.0")
                    if version != BLACKLIST_VERSION:
                        logger.warning(f"Blacklist version mismatch: expected {BLACKLIST_VERSION}, got {version}")

                    # Parse entries
                    blacklist_data = data.get("blacklist", {})
                    blacklist = {}
                    for hotkey, entry_data in blacklist_data.items():
                        try:
                            blacklist[hotkey] = BlacklistEntry(**entry_data)
                        except Exception as e:
                            logger.warning(f"Failed to parse blacklist entry for {hotkey}: {e}")

                    # Update cache
                    self._cache = blacklist
                    self._cache_timestamp = now

                    # Save to local cache
                    BLACKLIST_CACHE.write_bytes(body)
                    logger.info(f"Loaded {len(blacklist)} blacklisted miners from R2")

                    return blacklist

            except Exception as e:
                logger.warning(f"Failed to load blacklist from R2: {e}")

                # Try loading from local cache
                if BLACKLIST_CACHE.exists():
                    try:
                        data = orjson.loads(BLACKLIST_CACHE.read_bytes())
                        blacklist_data = data.get("blacklist", {})
                        blacklist = {}
                        for hotkey, entry_data in blacklist_data.items():
                            try:
                                blacklist[hotkey] = BlacklistEntry(**entry_data)
                            except Exception as e:
                                logger.warning(f"Failed to parse cached blacklist entry for {hotkey}: {e}")

                        self._cache = blacklist
                        self._cache_timestamp = now
                        logger.info(f"Loaded {len(blacklist)} blacklisted miners from local cache")
                        return blacklist
                    except Exception as e2:
                        logger.error(f"Failed to load blacklist from local cache: {e2}")

                # Return empty blacklist if all fails
                logger.warning("Using empty blacklist")
                self._cache = {}
                self._cache_timestamp = now
                return {}

    async def add_to_blacklist(
        self,
        hotkey: str,
        model: str,
        reason: str,
        similarity_score: float,
        block: int,
        samples_tested: int,
        details: Optional[Dict] = None,
    ) -> None:
        """Add a miner to the blacklist.

        Args:
            hotkey: Miner hotkey
            model: Model name
            reason: Reason for blacklisting
            similarity_score: Similarity score that triggered blacklisting
            block: Block number
            samples_tested: Number of samples tested
            details: Additional details
        """
        async with self._lock:
            # Load current blacklist
            blacklist = await self.load_blacklist(force_refresh=True)

            # Create new entry
            entry = BlacklistEntry(
                hotkey=hotkey,
                model=model,
                reason=reason,
                similarity_score=similarity_score,
                block=block,
                timestamp=time.time(),
                samples_tested=samples_tested,
                details=details or {},
            )

            # Add to blacklist
            blacklist[hotkey] = entry

            # Save to R2
            await self._save_blacklist(blacklist)

            logger.info(f"Added miner to blacklist: hotkey={hotkey}, model={model}, score={similarity_score:.4f}")

    async def remove_from_blacklist(self, hotkey: str) -> bool:
        """Remove a miner from the blacklist.

        Args:
            hotkey: Miner hotkey

        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            # Load current blacklist
            blacklist = await self.load_blacklist(force_refresh=True)

            if hotkey not in blacklist:
                logger.warning(f"Hotkey not found in blacklist: {hotkey}")
                return False

            # Remove from blacklist
            del blacklist[hotkey]

            # Save to R2
            await self._save_blacklist(blacklist)

            logger.info(f"Removed miner from blacklist: hotkey={hotkey}")
            return True

    async def is_blacklisted(self, hotkey: str) -> bool:
        """Check if a miner is blacklisted.

        Args:
            hotkey: Miner hotkey

        Returns:
            True if blacklisted, False otherwise
        """
        blacklist = await self.load_blacklist()
        return hotkey in blacklist

    async def get_blacklisted_hotkeys(self) -> Set[str]:
        """Get set of all blacklisted hotkeys.

        Returns:
            Set of blacklisted hotkeys
        """
        blacklist = await self.load_blacklist()
        return set(blacklist.keys())

    async def _save_blacklist(self, blacklist: Dict[str, BlacklistEntry]) -> None:
        """Save blacklist to R2.

        Args:
            blacklist: Blacklist dictionary
        """
        # Build JSON structure
        data = {
            "version": BLACKLIST_VERSION,
            "updated_at": int(time.time()),
            "blacklist": {
                hotkey: entry.to_dict()
                for hotkey, entry in blacklist.items()
            },
        }

        # Upload to R2
        body = orjson.dumps(data, option=orjson.OPT_INDENT_2)

        async with get_client_ctx() as client:
            await client.put_object(
                Bucket=FOLDER,
                Key=BLACKLIST_KEY,
                Body=body,
                ContentType="application/json",
            )

        # Update cache
        self._cache = blacklist
        self._cache_timestamp = time.time()

        # Save to local cache
        BLACKLIST_CACHE.write_bytes(body)

        logger.debug(f"Saved blacklist with {len(blacklist)} entries to R2")
