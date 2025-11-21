"""
Migration script to transfer data from R2 to DynamoDB

Handles incremental migration of historical sampling results.
Directly accesses R2 public storage without external dependencies.
"""

import os
import json
import time
import asyncio
import logging
import aiohttp
import orjson
from pathlib import Path
from typing import AsyncIterator, List, Dict, Any

from affine.database import init_client, close_client
from affine.database.dao import SampleResultsDAO

# Setup logging
from affine.core.setup import logger

# ============================================================================
# R2 Storage Configuration
# ============================================================================

# R2 public read configuration
R2_PUBLIC_BASE = "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev"
INDEX_KEY = "affine/index.json"
WINDOW = 20  # Block window size

# Cache configuration
CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR",
                 Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# R2 Storage Access Functions (Public Read Only)
# ============================================================================

def _w(b: int) -> int:
    """Calculate window-aligned block number."""
    return (b // WINDOW) * WINDOW


async def _get_http_session() -> aiohttp.ClientSession:
    """Get or create a global HTTP session for R2 access."""
    if not hasattr(_get_http_session, '_session'):
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        _get_http_session._session = aiohttp.ClientSession(timeout=timeout)
    return _get_http_session._session


async def _close_http_session():
    """Close the global HTTP session."""
    if hasattr(_get_http_session, '_session'):
        await _get_http_session._session.close()
        delattr(_get_http_session, '_session')


async def _load_public_index(need_blocks: set[int]) -> List[str]:
    """Load and filter keys from public R2 index.
    
    Args:
        need_blocks: Set of block numbers needed
    
    Returns:
        List of S3 keys matching the needed blocks
    """
    session = await _get_http_session()
    url = f"{R2_PUBLIC_BASE}/{INDEX_KEY}"
    
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        index_data = json.loads(await resp.text())
    
    # Filter keys by block number
    filtered_keys = []
    for key in index_data:
        # Extract block number from key (format: affine/results/{block}-{hotkey}.json)
        filename = Path(key).name
        block_str = filename.split("-", 1)[0]
        
        if block_str.isdigit() and int(block_str) in need_blocks:
            filtered_keys.append(key)
    
    return filtered_keys


async def _cache_shard(
    key: str,
    sem: asyncio.Semaphore,
    force_refresh: bool = False
) -> Path:
    """Download and cache a shard from R2 public storage.
    
    Args:
        key: S3 key to fetch
        sem: Semaphore for concurrency control
        force_refresh: If True, always download fresh copy
    
    Returns:
        Path to cached JSONL file
    """
    filename = Path(key).name
    cache_file = CACHE_DIR / f"{filename}.jsonl"
    mod_file = cache_file.with_suffix(".modified")
    
    # Check if cached version is valid
    if not force_refresh and cache_file.exists() and mod_file.exists():
        if cache_file.stat().st_size > 0:
            logger.debug(f"Using cached shard: {filename}")
            return cache_file
        else:
            logger.warning(f"Cached shard {filename} is empty, re-downloading")
    
    # Download with retry logic
    max_retries = 5
    base_delay = 5.0
    
    for attempt in range(max_retries):
        try:
            async with sem:
                session = await _get_http_session()
                url = f"{R2_PUBLIC_BASE}/{key}"
                
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    body = await resp.read()
                    last_modified = resp.headers.get("last-modified", str(time.time()))
                
                # Parse JSON array and convert to JSONL format
                data_array = orjson.loads(body)
                
                # Write to temporary file first
                tmp_file = cache_file.with_suffix(".tmp")
                with tmp_file.open("wb") as f:
                    for item in data_array:
                        f.write(orjson.dumps(item) + b"\n")
                
                # Atomic replace
                os.replace(tmp_file, cache_file)
                mod_file.write_text(last_modified)
                
                logger.debug(f"Downloaded shard: {filename} ({len(data_array)} records)")
                return cache_file
                
        except aiohttp.ClientResponseError as e:
            if e.status == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limited for {key}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.error(f"Error downloading {key}, retrying in {delay}s")
                await asyncio.sleep(delay)
            else:
                raise


def _jsonl_reader(path: Path):
    """Synchronous JSONL reader for maximum performance.
    
    Uses standard file I/O for better throughput with large files.
    """
    with open(path, "rb") as f:
        for line in f:
            line = line.rstrip(b"\n")
            if line:  # Skip empty lines
                yield line


async def load_r2_dataset(
    tail_blocks: int,
    current_block: int,
    max_concurrency: int = 5,
    refresh_window: int = 800
) -> AsyncIterator[Dict[str, Any]]:
    """Load dataset from R2 public storage.
    
    Args:
        tail_blocks: Number of blocks to look back
        current_block: Current blockchain block number
        max_concurrency: Maximum concurrent downloads
        refresh_window: Number of recent blocks to force refresh
    
    Yields:
        Dict representing each result record
    """
    # Calculate needed block windows
    need_blocks = {
        w for w in range(_w(current_block - tail_blocks), _w(current_block) + WINDOW, WINDOW)
    }
    
    logger.info(f"Loading R2 data for {len(need_blocks)} block windows")
    
    # Load index and filter keys
    keys = await _load_public_index(need_blocks)
    keys.sort()
    
    logger.info(f"Found {len(keys)} shards to process")
    
    if not keys:
        logger.warning("No data found in R2 public storage")
        return
    
    # Determine refresh threshold
    refresh_threshold = current_block - refresh_window
    sem = asyncio.Semaphore(max_concurrency)
    
    # Prefetch shards
    async def _prefetch_shard(key: str) -> Path | None:
        try:
            # Check if this block needs refresh
            block_str = Path(key).name.split("-", 1)[0]
            if block_str.isdigit():
                block_num = int(block_str)
                force_refresh = block_num >= refresh_threshold
            else:
                force_refresh = True
            
            return await _cache_shard(key, sem, force_refresh=force_refresh)
        except Exception as e:
            logger.warning(f"Failed to fetch shard {key}: {e}")
            return None
    
    # Create prefetch tasks
    tasks = [asyncio.create_task(_prefetch_shard(k)) for k in keys]
    
    # Process shards as they complete
    total_records = 0
    for coro in asyncio.as_completed(tasks):
        shard_path = await coro
        if shard_path is None:
            continue
        
        # Read JSONL file synchronously for performance
        shard_records = 0
        for raw_line in _jsonl_reader(shard_path):
            try:
                record = orjson.loads(raw_line)
                total_records += 1
                shard_records += 1
                
                # Log progress
                if total_records % 10000 == 0:
                    logger.info(f"Loaded {total_records} records from R2")
                
                yield record
                
            except Exception as e:
                logger.debug(f"Skipping invalid record: {e}")
        
        logger.debug(f"Processed shard {shard_path.name}: {shard_records} records")
    
    logger.info(f"R2 dataset loading complete: {total_records} total records")


# ============================================================================
# Migration Logic
# ============================================================================

class R2ToDynamoMigration:
    """Migrate data from R2 storage to DynamoDB."""
    
    def __init__(self):
        self.sample_dao = SampleResultsDAO()
        
        self.stats = {
            'total_processed': 0,
            'total_migrated': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'miners_updated': set(),
            'start_time': time.time()
        }
    
    async def migrate_result(self, result_dict: Dict[str, Any], deduplicate: bool = True) -> bool:
        """Migrate a single result to DynamoDB.
        
        Args:
            result_dict: Result data as dictionary
            deduplicate: If True, skip if already exists
            
        Returns:
            True if migrated successfully, False if skipped/error
        """
        try:
            # Extract fields from dict (avoiding class dependencies)
            miner_data = result_dict.get('miner', {})
            hotkey = miner_data.get('hotkey', '')
            revision = miner_data.get('revision') or 'unknown'
            model = miner_data.get('model') or 'unknown'
            block = miner_data.get('block') or 0
            
            env = result_dict.get('env', '')
            score = result_dict.get('score', 0.0)
            latency_seconds = result_dict.get('latency_seconds', 0.0)
            extra = result_dict.get('extra', {})
            timestamp_float = result_dict.get('timestamp', time.time())
            validator_hotkey = result_dict.get('hotkey', '')
            signature = result_dict.get('signature', '')
            
            # Extract task_id from extra.request.task_id or use direct task_id
            task_id = result_dict.get('task_id')
            if task_id is None and extra:
                request = extra.get('request', {})
                if isinstance(request, dict):
                    task_id = request.get('task_id')
            
            # Fallback to 'legacy' if still None
            if task_id is None:
                task_id = 'legacy'
            
            timestamp_ms = int(timestamp_float * 1000)
            
            # Check if already exists (deduplication)
            if deduplicate:
                existing = await self._check_sample_exists(
                    miner_hotkey=hotkey,
                    model_revision=revision,
                    env=env,
                    task_id=task_id,
                    timestamp=timestamp_ms
                )
                if existing:
                    self.stats['total_skipped'] += 1
                    return False
            
            # Save sample result
            await self.sample_dao.save_sample(
                miner_hotkey=hotkey,
                model_revision=revision,
                model=model,
                env=env,
                task_id=task_id,
                score=score,
                latency_ms=int(latency_seconds * 1000),
                extra=extra,
                validator_hotkey=validator_hotkey,
                block_number=block,
                signature=signature,
                timestamp=timestamp_ms
            )
            
            # Track miners updated
            self.stats['miners_updated'].add(hotkey)
            self.stats['total_migrated'] += 1
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error migrating result: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def _check_sample_exists(
        self,
        miner_hotkey: str,
        model_revision: str,
        env: str,
        task_id: str,
        timestamp: int
    ) -> bool:
        """Check if a sample already exists in DynamoDB.
        
        Uses PK/SK to find the item, then verifies timestamp matches.
        This prevents false positives when the same task_id is executed
        multiple times (e.g., retries).
        
        Returns:
            True if sample exists with matching timestamp
        """
        try:
            pk = self.sample_dao._make_pk(miner_hotkey, model_revision, env)
            sk = self.sample_dao._make_sk(task_id)
            
            from affine.database.client import get_client
            client = get_client()
            
            response = await client.get_item(
                TableName=self.sample_dao.table_name,
                Key={
                    'pk': {'S': pk},
                    'sk': {'S': sk}
                }
            )
            
            if 'Item' not in response:
                return False
            
            # Verify timestamp matches to handle task retries
            item = self.sample_dao._deserialize(response['Item'])
            existing_timestamp = item.get('timestamp')
            
            # Consider it a duplicate if timestamps are within 1 second (1000ms)
            if existing_timestamp and abs(existing_timestamp - timestamp) < 1000:
                return True
            
            return False
            
        except Exception:
            # On error, assume doesn't exist (will try to insert)
            return False
    
    async def migrate_batch(self, results: List[Dict[str, Any]], batch_size: int = 100):
        """Migrate a batch of results.
        
        Args:
            results: List of result dicts to migrate
            batch_size: Number of results to process concurrently
        """
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            
            tasks = [self.migrate_result(r) for r in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.stats['total_processed'] += len(batch)
            
            # Print progress
            if self.stats['total_processed'] % 1000 == 0:
                elapsed = time.time() - self.stats['start_time']
                rate = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {self.stats['total_processed']} results ({rate:.1f} results/sec)")
    
    async def migrate_from_r2(
        self,
        tail_blocks: int = 100000,
        current_block: int = None,
        batch_size: int = 100,
        max_results: int = None
    ):
        """Migrate data from R2 storage.
        
        Args:
            tail_blocks: Number of blocks to look back
            current_block: Current block number (if None, fetched from chain)
            batch_size: Batch size for concurrent processing
            max_results: Maximum number of results to migrate (None = all)
        """
        # Get current block if not provided
        if current_block is None:
            logger.info("Fetching current block from chain...")
            from affine.utils.subtensor import get_subtensor
            sub = await get_subtensor()
            current_block = await sub.get_current_block()
        
        logger.info(f"Starting migration from R2 (tail={tail_blocks} blocks, current_block={current_block})")
        
        results_buffer = []
        count = 0
        
        try:
            async for result_dict in load_r2_dataset(tail_blocks, current_block):
                results_buffer.append(result_dict)
                count += 1
                
                # Process batch
                if len(results_buffer) >= batch_size:
                    await self.migrate_batch(results_buffer, batch_size)
                    results_buffer = []
                
                # Check max limit
                if max_results and count >= max_results:
                    logger.info(f"Reached max_results limit: {max_results}")
                    break
            
            # Process remaining results
            if results_buffer:
                await self.migrate_batch(results_buffer, batch_size)
            
        finally:
            # Close HTTP session
            await _close_http_session()
        
        self.print_summary()
    
    def print_summary(self):
        """Print migration summary."""
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['total_processed'] / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("Migration Summary")
        print("="*60)
        print(f"Total Processed:  {self.stats['total_processed']}")
        print(f"Total Migrated:   {self.stats['total_migrated']}")
        print(f"Total Skipped:    {self.stats['total_skipped']}")
        print(f"Total Errors:     {self.stats['total_errors']}")
        print(f"Miners Updated:   {len(self.stats['miners_updated'])}")
        print(f"Elapsed Time:     {elapsed:.1f}s")
        print(f"Migration Rate:   {rate:.1f} results/sec")
        print("="*60)


# ============================================================================
# CLI Entry Point
# ============================================================================

async def run_migration(
    tail_blocks: int = 100000,
    current_block: int = None,
    max_results: int = None
):
    """Run the migration process.
    
    Args:
        tail_blocks: Number of blocks to look back
        current_block: Current block number (if None, fetched from chain)
        max_results: Maximum number of results to migrate
    """
    # Initialize DynamoDB client
    await init_client()
    
    try:
        migration = R2ToDynamoMigration()
        await migration.migrate_from_r2(
            tail_blocks=tail_blocks,
            current_block=current_block,
            max_results=max_results
        )
    finally:
        # Close DynamoDB client
        await close_client()


if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Migrate data from R2 to DynamoDB")
    parser.add_argument(
        "--tail",
        type=int,
        default=100000,
        help="Number of blocks to look back (default: 100000)"
    )
    parser.add_argument(
        "--current-block",
        type=int,
        default=None,
        help="Current block number (default: fetch from chain)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum number of results to migrate (default: all)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_migration(
        tail_blocks=args.tail,
        current_block=args.current_block,
        max_results=args.max_results
    ))