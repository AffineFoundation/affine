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
from affine.query import _get_client
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

async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    name, out = Path(key).name, None
    out = CACHE_DIR / f"{name}.jsonl"; mod = out.with_suffix(".modified")
    max_retries = 5
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            async with sem:
                if PUBLIC_READ:
                    sess = await _get_client()
                    url = f"{R2_PUBLIC_BASE}/{key}"
                    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
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

async def _jsonl(path: Path) -> AsyncIterator[bytes]:
    async with aiofiles.open(path, "rb") as f:
        async for line in f:
            yield line.rstrip(b"\n")

async def dataset(
    tail: int,
    *,
    max_concurrency: int = 10,
    compact: bool = True,
) -> AsyncIterator["Result | CompactResult"]:
    """Load dataset from R2 storage.
    
    Args:
        tail: Number of blocks to look back
        max_concurrency: Maximum concurrent downloads
        compact: If True, return CompactResult (memory-efficient). If False, return full Result.
    
    Yields:
        CompactResult or Result objects depending on compact flag
    """
    sub = await get_subtensor()
    cur = await sub.get_current_block()
    need = {w for w in range(_w(cur - tail), _w(cur) + WINDOW, WINDOW)}
    keys = [
        k for k in await _index()
        if (h := Path(k).name.split("-", 1)[0]).isdigit() and int(h) in need
    ]
    keys.sort()
    sem = asyncio.Semaphore(max_concurrency)
    
    async def _prefetch(key: str) -> Path | None:
        try:
            return await _cache_shard(key, sem)
        except Exception:
            import traceback
            traceback.print_exc()
            logger.warning(f"Failed to fetch key: {key}, skipping")
            return None
    
    bar = tqdm(desc=f"Dataset=({cur}, {cur - tail})", unit="res", dynamic_ncols=True)
    try:
        tasks = [asyncio.create_task(_prefetch(k)) for k in keys]

        for coro in asyncio.as_completed(tasks):
            path = await coro
            if path is None:
                continue

            async for raw in _jsonl(path):
                try:
                    if compact:
                        # Parse only the fields needed for weight calculation
                        data = orjson.loads(raw)
                        
                        # Verify signature before converting to compact format
                        r = Result.model_validate(data)
                        if r.verify():
                            compact_result = CompactResult.from_result(r)
                            bar.update(1)
                            yield compact_result
                            # Explicitly delete full result to free memory immediately
                            del r
                    else:
                        # Return full Result object (legacy mode)
                        r = Result.model_validate(orjson.loads(raw))
                        if r.verify():
                            bar.update(1)
                            yield r
                except Exception as e:
                    # Skip invalid results but log for debugging
                    logger.debug(f"Skipping invalid result: {type(e).__name__}: {e}")
    finally:
        bar.close()

async def sign_results( wallet, results ):
    try:
        signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
        timeout = aiohttp.ClientTimeout(connect=2, total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payloads = [str(r.challenge) for r in results]
            resp = await session.post(f"{signer_url}/sign", json={"payloads": payloads})
            if resp.status == 200:
                data = await resp.json()
                sigs = data.get("signatures") or []
                hotkey = data.get("hotkey")
                for r, s in zip(results, sigs):
                    r.hotkey = hotkey
                    r.signature = s
    except Exception as e:
        logger.info(f"sink: signer unavailable, using local signing: {type(e).__name__}: {e}")
        hotkey = wallet.hotkey.ss58_address
        for r in results: 
            r.sign(wallet)
    finally:
        return hotkey, results

SINK_BUFFER: list["Result"] = []
SINK_BUFFER_SIZE = int(os.getenv("AFFINE_SINK_BUFFER", "100"))

async def sink_enqueue(wallet, block, results, force: bool = False):
    global SINK_BUFFER
    SINK_BUFFER.extend(results)
    if not force and len(SINK_BUFFER) < SINK_BUFFER_SIZE: return
    buf, SINK_BUFFER = SINK_BUFFER, []
    await sink(wallet=wallet, results=buf, block=block)

async def sink(wallet, results: list["Result"], block: int = None):
    if not results: return
    if block is None:
        sub = await get_subtensor(); block = await sub.get_current_block()
    hotkey, signed = await sign_results( wallet, results )
    key = f"{RESULT_PREFIX}{_w(block):09d}-{hotkey}.json"
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