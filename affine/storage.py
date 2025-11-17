import os
import json
import time
import asyncio
import logging
import aiohttp
import orjson
import aiofiles
from pathlib import Path
from typing import AsyncIterator
from tqdm.asyncio import tqdm

from aiobotocore.session import get_session
from botocore.config import Config

from affine.config import get_conf
from affine.utils.subtensor import get_subtensor
from affine.models import Result, CompactResult
from affine.http_client import _get_client
from affine.setup import logger

import numpy as np

WINDOW        = int(os.getenv("AFFINE_WINDOW", 20))
RESULT_PREFIX = "affine/results/"
INDEX_KEY     = "affine/index.json"

FOLDER  = os.getenv("R2_FOLDER", "affine" )
BUCKET  = os.getenv("R2_BUCKET_ID", "00523074f51300584834607253cae0fa" )
ACCESS  = os.getenv("R2_WRITE_ACCESS_KEY_ID", "")
SECRET  = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

PUBLIC_READ = os.getenv("AFFINE_R2_PUBLIC", "1") == "1"
ACCOUNT_ID  = os.getenv("R2_ACCOUNT_ID", BUCKET)

R2_PUBLIC_BASE = f"https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev"

get_client_ctx = lambda: get_session().create_client(
    "s3", endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
    config=Config(max_pool_connections=256)
)

CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR",
                 Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _w(b: int) -> int: return (b // WINDOW) * WINDOW

async def _index() -> list[str]:
    if PUBLIC_READ:
        sess = await _get_client()
        url = f"{R2_PUBLIC_BASE}/{INDEX_KEY}"
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            return json.loads(await r.text())
    async with get_client_ctx() as c:
        r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
        return json.loads(await r["Body"].read())

async def _update_index(k: str) -> None:
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
            idx = set(json.loads(await r["Body"].read()))
        except c.exceptions.NoSuchKey:
            idx = set()
        if k not in idx:
            idx.add(k)
            await c.put_object(Bucket=FOLDER, Key=INDEX_KEY,
                               Body=orjson.dumps(sorted(idx)),
                               ContentType="application/json")

async def _cache_shard(key: str, sem: asyncio.Semaphore, use_public: bool = None, force_refresh: bool = False) -> Path:
    """Cache a shard from R2 storage.
    
    Args:
        key: S3 key to fetch
        sem: Semaphore for concurrency control
        use_public: If True, force public read. If False, force private read. If None, use PUBLIC_READ global.
        force_refresh: If True, always download fresh copy. If False, use cached version if available.
    """
    name, out = Path(key).name, None
    out = CACHE_DIR / f"{name}.jsonl"; mod = out.with_suffix(".modified")
    max_retries = 5
    base_delay = 5.0
    
    # Determine which read mode to use
    read_public = PUBLIC_READ if use_public is None else use_public
    
    # Check if we can use cached version (skip download if cache exists and not forcing refresh)
    if not force_refresh and out.exists() and mod.exists():
        # Verify cache file is not empty before trusting it
        if out.stat().st_size > 0:
            logger.debug(f"Using cached shard: {name}")
            return out
        else:
            logger.warning(f"Cached shard {name} is empty, will re-download")
    
    for attempt in range(max_retries):
        try:
            async with sem:
                if read_public:
                    sess = await _get_client()
                    url = f"{R2_PUBLIC_BASE}/{key}"
                    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=300)) as r:
                        r.raise_for_status()
                        body = await r.read()
                        lm = r.headers.get("last-modified", str(time.time()))
                    tmp = out.with_suffix(".tmp")
                    with tmp.open("wb") as f:
                        f.write(b"\n".join(orjson.dumps(i) for i in orjson.loads(body)) + b"\n")
                    os.replace(tmp, out); mod.write_text(lm)
                    return out
                async with get_client_ctx() as c:
                    if out.exists() and mod.exists():
                        h = await c.head_object(Bucket=FOLDER, Key=key)
                        if h["LastModified"].isoformat() == mod.read_text().strip():
                            return out
                    o = await c.get_object(Bucket=FOLDER, Key=key)
                    body, lm = await o["Body"].read(), o["LastModified"].isoformat()
            tmp = out.with_suffix(".tmp")
            with tmp.open("wb") as f:
                f.write(b"\n".join(orjson.dumps(i) for i in orjson.loads(body)) + b"\n")
            os.replace(tmp, out); mod.write_text(lm)
            return out
        except aiohttp.ClientResponseError as e:
            if e.status == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                if attempt > 1:
                    logger.warning(f"Rate limited for key: {key}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception:
            raise

def _jsonl_sync(path: Path):
    """Synchronous JSONL reader for maximum performance.
    
    Uses standard file I/O instead of aiofiles for better throughput.
    For 2M+ records, sync I/O is actually faster than async line-by-line.
    """
    with open(path, "rb") as f:
        for line in f:
            yield line.rstrip(b"\n")


async def _jsonl(path: Path) -> AsyncIterator[bytes]:
    async with aiofiles.open(path, "rb") as f:
        async for line in f:
            yield line.rstrip(b"\n")

async def _load_public_index(need: set[int]) -> list[str]:
    """Load and filter keys from public repository index.
    
    Args:
        need: Set of block numbers needed
    
    Returns:
        List of keys matching the needed blocks
    """
    sess = await _get_client()
    url = f"{R2_PUBLIC_BASE}/{INDEX_KEY}"
    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
        r.raise_for_status()
        public_index = json.loads(await r.text())
    
    return [
        k for k in public_index
        if (h := Path(k).name.split("-", 1)[0]).isdigit() and int(h) in need
    ]

async def dataset(
    tail: int,
    *,
    max_concurrency: int = 5,
    compact: bool = True,
    refresh_window: int = 800,
) -> AsyncIterator["Result | CompactResult"]:
    """Load dataset from R2 storage.
    
    Automatically falls back to public repository if own repository has insufficient samples.
    Uses intelligent caching: only refreshes recent blocks (within refresh_window), reuses older cached data.
    
    Args:
        tail: Number of blocks to look back
        max_concurrency: Maximum concurrent downloads
        compact: If True, return CompactResult (memory-efficient). If False, return full Result.
        refresh_window: Number of recent blocks to force refresh (default 1000). Older blocks use cache.
    
    Yields:
        CompactResult or Result objects depending on compact flag
    """
    sub = await get_subtensor()
    cur = await sub.get_current_block()
    need = {w for w in range(_w(cur - tail), _w(cur) + WINDOW, WINDOW)}
    refresh_threshold = cur - refresh_window
    
    use_public_fallback = False
    keys = []
    
    # Determine data source based on PUBLIC_READ setting
    if PUBLIC_READ:
        # Already in public mode, use public repository directly
        logger.info("Using public repository (PUBLIC_READ=True)")
        use_public_fallback = True
        
        try:
            keys = await _load_public_index(need)
            logger.info(f"Loaded {len(keys)} keys from public repository")
        except Exception as e:
            logger.error(f"Failed to load from public repository: {e}")
            keys = []
    else:
        # PUBLIC_READ=False: try private repository first
        try:
            async def _get_index_private():
                async with get_client_ctx() as c:
                    r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
                    return json.loads(await r["Body"].read())
            
            own_index = await _get_index_private()
            keys = [
                k for k in own_index
                if (h := Path(k).name.split("-", 1)[0]).isdigit() and int(h) in need
            ]
            logger.info(f"Found {len(keys)} keys in private repository")
            
            # Check if private repository has sufficient data
            if len(keys) < 50:
                logger.info(f"Insufficient keys in private repository ({len(keys)} keys), falling back to public")
                use_public_fallback = True
                
                # Fall back to public repository
                try:
                    keys = await _load_public_index(need)
                    logger.info(f"Loaded {len(keys)} keys from public repository (fallback)")
                except Exception as e:
                    logger.error(f"Failed to load from public repository during fallback: {e}")
                    # Keep using private keys if public fallback fails
                    use_public_fallback = False
            
        except Exception as e:
            logger.warning(f"Failed to load from private repository: {e}, falling back to public")
            use_public_fallback = True
            
            # Fall back to public repository on private load failure
            try:
                keys = await _load_public_index(need)
                logger.info(f"Loaded {len(keys)} keys from public repository (fallback after error)")
            except Exception as e2:
                logger.error(f"Failed to load from public repository during fallback: {e2}")
                keys = []
    
    keys.sort()
    sem = asyncio.Semaphore(max_concurrency)
    
    async def _prefetch(key: str, use_public: bool) -> Path | None:
        try:
            # Determine if this block needs refresh
            block_num_str = Path(key).name.split("-", 1)[0]
            if block_num_str.isdigit():
                block_num = int(block_num_str)
                force_refresh = block_num >= refresh_threshold
            else:
                force_refresh = True  # Unknown block format, refresh to be safe
            
            return await _cache_shard(key, sem, use_public=use_public, force_refresh=force_refresh)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.warning(f"Failed to fetch key: {key}, skipping")
            return None
    
    bar = tqdm(desc=f"Dataset=({cur}, {cur - tail})", unit="res", dynamic_ncols=True)
    try:
        tasks = [asyncio.create_task(_prefetch(k, use_public_fallback)) for k in keys]

        for coro in asyncio.as_completed(tasks):
            path = await coro
            if path is None:
                continue
            
            # Extract block number from filename (format: {block}-{hotkey}.jsonl)
            block_str = path.stem.split('-')[0] if path else 'unknown'

            # Use synchronous reading for maximum performance
            # Async line-by-line reading has too much overhead for 2M+ records
            count = 0
            batch_size = 1000  # Update progress bar every 1000 records
            
            for raw in _jsonl_sync(path):
                try:
                    data = orjson.loads(raw)

                    # Fast path: skip Pydantic validation for performance (200万+ records)
                    if compact:
                        # Ultra-fast: direct attribute access without any validation
                        # Extract only the fields we need, inline
                        miner_data = data.get('miner', {})
                        task_id = data.get('task_id')
                        if task_id is None:
                            extra = data.get('extra', {})
                            if isinstance(extra, dict):
                                request = extra.get('request', {})
                                if isinstance(request, dict):
                                    task_id = request.get('task_id')
                        
                        # Use model_construct for zero-overhead instantiation
                        compact_result = CompactResult.model_construct(
                            hotkey=miner_data.get('hotkey', ''),
                            uid=miner_data.get('uid', 0),
                            model=miner_data.get('model'),
                            revision=miner_data.get('revision'),
                            block=miner_data.get('block'),
                            env=data.get('env', ''),
                            score=data.get('score', 0.0),
                            task_id=task_id,
                            timestamp=data.get('timestamp', 0.0)
                        )
                        count += 1
                        if count % batch_size == 0:
                            bar.update(batch_size)
                        yield compact_result
                    else:
                        # Use model_construct for zero-validation instantiation
                        r = Result.model_construct(**data)
                        count += 1
                        if count % batch_size == 0:
                            bar.update(batch_size)
                        yield r
                except Exception as e:
                    # Skip invalid results but log for debugging
                    logger.debug(f"Skipping invalid result: {type(e).__name__}: {e}")
            
            # Update remaining count
            remaining = count % batch_size
            if remaining > 0:
                bar.update(remaining)
    finally:
        bar.close()

async def sign_results(wallet, results: list["Result"]) -> tuple[str, list["Result"]]:
    signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
    try:
        logger.debug(f"Attempting remote signing via {signer_url} for {len(results)} results")
        
        timeout = aiohttp.ClientTimeout(connect=2, total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payloads = [r._get_sign_data() for r in results]

            resp = await session.post(f"{signer_url}/sign", json={"payloads": payloads})
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(
                    f"Signer service returned status {resp.status}: {error_text[:200]}"
                )
                raise RuntimeError(f"Signer returned status {resp.status}")

            data = await resp.json()
            signatures = data.get("signatures")
            hotkey = data.get("hotkey")
            
            if not signatures:
                logger.error("Signer response missing 'signatures' field")
                raise ValueError("Invalid signer response: missing signatures")
            
            if len(signatures) != len(results):
                logger.error(
                    f"Signature count mismatch: expected {len(results)}, got {len(signatures)}"
                )
                raise ValueError("Signature count mismatch")
            
            if not hotkey:
                logger.warning("Signer response missing 'hotkey' field, will use wallet fallback")
            else:
                # Apply signatures to results
                for result, signature in zip(results, signatures):
                    result.hotkey = hotkey
                    result.signature = signature
                
                logger.info(f"Successfully signed {len(results)} results remotely with hotkey {hotkey}")
                return hotkey, results
                
    except aiohttp.ClientConnectionError as e:
        logger.warning(f"Failed to connect to signer service at {signer_url}: {e}")
    except asyncio.TimeoutError:
        logger.warning(f"Signer service timeout after 30s at {signer_url}")
    except aiohttp.ClientResponseError as e:
        logger.error(f"Signer service HTTP error: status={e.status}, message={e.message}")
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Invalid signer response format: {type(e).__name__}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during remote signing: {type(e).__name__}: {e}")
    
    # Fallback to local wallet signing
    logger.info(f"Using local wallet signing for {len(results)} results")
    hotkey = wallet.hotkey.ss58_address
    
    for result in results:
        try:
            result.sign(wallet)
        except Exception as e:
            logger.error(f"Failed to sign result locally for miner {result.miner.uid}: {e}")
            raise
    
    logger.debug(f"Successfully signed {len(results)} results locally with hotkey {hotkey}")
    return hotkey, results

async def sink(wallet, results: list["Result"], block: int = None):
    if not results: return
    if block is None:
        sub = await get_subtensor(); block = await sub.get_current_block()
    hotkey, signed = await sign_results( wallet, results )
    key = f"{RESULT_PREFIX}{_w(block):09d}-{hotkey}.json"
    logger.debug(f"results key {key}")
    dumped = [ r.model_dump(mode="json") for r in signed ]
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=key)
            merged = json.loads(await r["Body"].read()) + dumped
        except c.exceptions.NoSuchKey:
            merged = dumped
        await c.put_object(Bucket=FOLDER, Key=key, Body=orjson.dumps(merged),
                           ContentType="application/json")
    if len(merged) == len(dumped):
        await _update_index(key)

async def prune(tail: int):
    """Prune old cache files that are older than the tail window.

    Args:
        tail: Number of blocks to keep in cache
    """
    sub = await get_subtensor()
    cur = await sub.get_current_block()
    for f in CACHE_DIR.glob("*.jsonl"):
        b = f.name.split("-", 1)[0]
        if b.isdigit() and int(b) < cur - tail:
            try:
                f.unlink()
            except OSError as e:
                logger.debug(f"Failed to delete cache file {f.name}: {e}")

WEIGHTS_PREFIX = "affine/weights/"
WEIGHTS_LATEST_KEY = "affine/weights/latest.json"
SUMMARY_SCHEMA_VERSION = "1.0.0"


def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types.

    Handles numpy integers, floats, arrays, and special values (NaN, Inf).
    Also handles nested dictionaries and lists.

    Args:
        obj: Object to convert (can be numpy type, dict, list, or primitive)

    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Handle special float values
        if np.isnan(val) or np.isinf(val):
            return None  # Convert NaN/Inf to None for JSON compatibility
        return val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

async def save_summary(block: int, summary_data: dict):
    """Save validator summary to S3 weights folder with flexible schema.

    Saves the summary to both a block-specific key and a latest.json key for fast access.

    Args:
        block: Current block number
        summary_data: Dictionary containing summary information with the following keys:
            - header: List of column names
            - rows: List of row data (can be list or dict)
            - miners: Optional dict mapping hotkey -> miner details
            - stats: Optional additional statistics
            - Any other custom fields
    """
    # Convert any numpy types to native Python types
    clean_data = _convert_numpy_types(summary_data)

    # Build flexible schema wrapper
    output = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "timestamp": int(time.time()),
        "block": int(block),  # Ensure block is also a native int
        "data": clean_data
    }

    key = f"{WEIGHTS_PREFIX}summary-{block}.json"
    body = orjson.dumps(output, option=orjson.OPT_INDENT_2)

    async with get_client_ctx() as c:
        # Upload to both block-specific and latest keys in parallel
        await asyncio.gather(
            c.put_object(
                Bucket=FOLDER,
                Key=key,
                Body=body,
                ContentType="application/json"
            ),
            c.put_object(
                Bucket=FOLDER,
                Key=WEIGHTS_LATEST_KEY,
                Body=body,
                ContentType="application/json"
            )
        )

    logger.info(f"Saved summary to S3: {key} and {WEIGHTS_LATEST_KEY} (schema v{SUMMARY_SCHEMA_VERSION})")

async def load_summary(block: int = None) -> dict:
    """Load validator summary from S3 weights folder.

    Args:
        block: Block number to load. If None, loads the latest available summary.

    Returns:
        Dictionary containing the summary data

    Raises:
        Exception: If summary file not found or cannot be loaded
    """
    if block is None:
        # Fast path: Read from latest.json (O(1) access, no listing needed)
        key = WEIGHTS_LATEST_KEY
        logger.info(f"Loading latest summary from {key}")
    else:
        # Specific block requested
        key = f"{WEIGHTS_PREFIX}summary-{block}.json"
        logger.info(f"Loading summary for block {block}")

    # Load the summary
    if PUBLIC_READ:
        sess = await _get_client()
        url = f"{R2_PUBLIC_BASE}/{key}"
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            data = json.loads(await r.text())
    else:
        async with get_client_ctx() as c:
            try:
                r = await c.get_object(Bucket=FOLDER, Key=key)
                data = json.loads(await r["Body"].read())
            except c.exceptions.NoSuchKey:
                raise FileNotFoundError(f"Summary not found: {key}")

    logger.info(f"Loaded summary from S3: {key}")
    return data