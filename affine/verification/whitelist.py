"""Whitelist management for verified miners."""

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

WHITELIST_KEY = "affine/verification/whitelist.json"
WHITELIST_VERSION = "1.0.0"

# Local cache
CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR", Path.home() / ".cache" / "affine" / "verification"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WHITELIST_CACHE = CACHE_DIR / "whitelist.json"

# Verification validity period (7 days in seconds)
VERIFICATION_VALIDITY = 7 * 24 * 60 * 60


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
class WhitelistEntry:
    """Whitelist entry for a verified miner."""

    hotkey: str
    model: str
    revision: str
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

    def is_valid(self, now: Optional[float] = None) -> bool:
        """Check if verification is still valid (within 7 days).

        Args:
            now: Current timestamp (default: time.time())

        Returns:
            True if verification is still valid
        """
        if now is None:
            now = time.time()
        return (now - self.timestamp) < VERIFICATION_VALIDITY


class WhitelistManager:
    """Manage whitelist for verified miners."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._cache: Optional[Dict[str, WhitelistEntry]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl = 300  # 5 minutes

    async def load_whitelist(self, force_refresh: bool = False) -> Dict[str, WhitelistEntry]:
        """Load whitelist from R2 or cache.

        Args:
            force_refresh: Force refresh from R2 even if cache is valid

        Returns:
            Dictionary mapping hotkey to WhitelistEntry
        """
        async with self._lock:
            now = time.time()

            # Return cached data if still valid
            if not force_refresh and self._cache is not None and (now - self._cache_timestamp) < self._cache_ttl:
                logger.debug("Using cached whitelist")
                return self._cache

            # Try loading from R2
            try:
                async with get_client_ctx() as client:
                    response = await client.get_object(Bucket=FOLDER, Key=WHITELIST_KEY)
                    body = await response["Body"].read()
                    data = orjson.loads(body)

                    # Validate version
                    version = data.get("version", "0.0.0")
                    if version != WHITELIST_VERSION:
                        logger.warning(f"Whitelist version mismatch: expected {WHITELIST_VERSION}, got {version}")

                    # Parse entries
                    whitelist_data = data.get("whitelist", {})
                    whitelist = {}
                    for hotkey, entry_data in whitelist_data.items():
                        try:
                            whitelist[hotkey] = WhitelistEntry(**entry_data)
                        except Exception as e:
                            logger.warning(f"Failed to parse whitelist entry for {hotkey}: {e}")

                    # Update cache
                    self._cache = whitelist
                    self._cache_timestamp = now

                    # Save to local cache
                    WHITELIST_CACHE.write_bytes(body)
                    logger.info(f"Loaded {len(whitelist)} verified miners from R2")

                    return whitelist

            except Exception as e:
                logger.warning(f"Failed to load whitelist from R2: {e}")

                # Try loading from local cache
                if WHITELIST_CACHE.exists():
                    try:
                        data = orjson.loads(WHITELIST_CACHE.read_bytes())
                        whitelist_data = data.get("whitelist", {})
                        whitelist = {}
                        for hotkey, entry_data in whitelist_data.items():
                            try:
                                whitelist[hotkey] = WhitelistEntry(**entry_data)
                            except Exception as e:
                                logger.warning(f"Failed to parse cached whitelist entry for {hotkey}: {e}")

                        self._cache = whitelist
                        self._cache_timestamp = now
                        logger.info(f"Loaded {len(whitelist)} verified miners from local cache")
                        return whitelist
                    except Exception as e2:
                        logger.error(f"Failed to load whitelist from local cache: {e2}")

                # Return empty whitelist if all fails
                logger.warning("Using empty whitelist")
                self._cache = {}
                self._cache_timestamp = now
                return {}

    async def add_to_whitelist(
        self,
        hotkey: str,
        model: str,
        revision: str,
        similarity_score: float,
        block: int,
        samples_tested: int,
        details: Optional[Dict] = None,
    ) -> None:
        """Add a verified miner to the whitelist.

        Args:
            hotkey: Miner hotkey
            model: Model name
            revision: Model revision
            similarity_score: Similarity score
            block: Block number
            samples_tested: Number of samples tested
            details: Additional details
        """
        async with self._lock:
            # Load current whitelist
            whitelist = await self.load_whitelist(force_refresh=True)

            # Create new entry
            entry = WhitelistEntry(
                hotkey=hotkey,
                model=model,
                revision=revision,
                similarity_score=similarity_score,
                block=block,
                timestamp=time.time(),
                samples_tested=samples_tested,
                details=details or {},
            )

            # Add to whitelist
            whitelist[hotkey] = entry

            # Save to R2
            await self._save_whitelist(whitelist)

            logger.info(f"Added miner to whitelist: hotkey={hotkey}, model={model}, score={similarity_score:.4f}")

    async def remove_from_whitelist(self, hotkey: str) -> bool:
        """Remove a miner from the whitelist.

        Args:
            hotkey: Miner hotkey

        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            # Load current whitelist
            whitelist = await self.load_whitelist(force_refresh=True)

            if hotkey not in whitelist:
                logger.warning(f"Hotkey not found in whitelist: {hotkey}")
                return False

            # Remove from whitelist
            del whitelist[hotkey]

            # Save to R2
            await self._save_whitelist(whitelist)

            logger.info(f"Removed miner from whitelist: hotkey={hotkey}")
            return True

    async def is_verified(self, hotkey: str, model: str, revision: str) -> bool:
        """Check if a miner's model is verified and still valid.

        Args:
            hotkey: Miner hotkey
            model: Model name
            revision: Model revision

        Returns:
            True if verified and valid, False otherwise
        """
        whitelist = await self.load_whitelist()

        if hotkey not in whitelist:
            return False

        entry = whitelist[hotkey]

        # Check if verification is still valid (within 7 days)
        if not entry.is_valid():
            logger.debug(f"Verification expired for {hotkey} (age: {time.time() - entry.timestamp:.0f}s)")
            return False

        # Check if model and revision match
        if entry.model != model or entry.revision != revision:
            logger.debug(f"Model or revision mismatch for {hotkey}: "
                        f"cached=({entry.model}, {entry.revision}), current=({model}, {revision})")
            return False

        return True

    async def get_verified_hotkeys(self) -> Set[str]:
        """Get set of all verified hotkeys (including expired ones).

        Returns:
            Set of verified hotkeys
        """
        whitelist = await self.load_whitelist()
        return set(whitelist.keys())

    async def _save_whitelist(self, whitelist: Dict[str, WhitelistEntry]) -> None:
        """Save whitelist to R2.

        Args:
            whitelist: Whitelist dictionary
        """
        # Build JSON structure
        data = {
            "version": WHITELIST_VERSION,
            "updated_at": int(time.time()),
            "whitelist": {
                hotkey: entry.to_dict()
                for hotkey, entry in whitelist.items()
            },
        }

        # Upload to R2
        body = orjson.dumps(data, option=orjson.OPT_INDENT_2)

        async with get_client_ctx() as client:
            await client.put_object(
                Bucket=FOLDER,
                Key=WHITELIST_KEY,
                Body=body,
                ContentType="application/json",
            )

        # Update cache
        self._cache = whitelist
        self._cache_timestamp = time.time()

        # Save to local cache
        WHITELIST_CACHE.write_bytes(body)

        logger.debug(f"Saved whitelist with {len(whitelist)} entries to R2")
