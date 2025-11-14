"""
Migration script to transfer data from R2 to DynamoDB

Handles incremental migration of historical sampling results.
"""

import asyncio
import json
import time
from typing import List, Dict, Any
from pathlib import Path

from affine.storage import dataset
from affine.models import Result
from affine.database import init_client, close_client
from affine.database.dao import SampleResultsDAO, MinerMetadataDAO


class R2ToDynamoMigration:
    """Migrate data from R2 storage to DynamoDB."""
    
    def __init__(self):
        self.sample_dao = SampleResultsDAO()
        self.miner_dao = MinerMetadataDAO()
        
        self.stats = {
            'total_processed': 0,
            'total_migrated': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'miners_updated': set(),
            'start_time': time.time()
        }
    
    async def migrate_result(self, result: Result) -> bool:
        """Migrate a single result to DynamoDB.
        
        Args:
            result: Result object from R2
            
        Returns:
            True if migrated successfully
        """
        try:
            # Extract conversation data from extra field
            conversation = result.extra.get('conversation', {})
            
            # Save sample result
            await self.sample_dao.save_sample(
                miner_hotkey=result.miner.hotkey,
                model_revision=result.miner.revision or 'unknown',
                model_hash=result.miner.model or 'unknown',
                uid=result.miner.uid,
                env=result.env,
                subset=result.extra.get('subset', 'unknown'),
                dataset_index=result.extra.get('dataset_index', 0),
                task_id=result.extra.get('task_id', 'legacy'),
                score=result.score,
                success=result.success,
                latency_ms=int(result.latency_seconds * 1000),
                conversation=conversation,
                validator_hotkey=result.hotkey,
                block_number=result.miner.block or 0,
                signature=result.signature,
                timestamp=int(result.timestamp * 1000)
            )
            
            # Update miner metadata
            self.stats['miners_updated'].add(result.miner.hotkey)
            
            # Update miner metadata if not exists
            existing_metadata = await self.miner_dao.get_metadata(result.miner.hotkey)
            if not existing_metadata:
                await self.miner_dao.save_metadata(
                    miner_hotkey=result.miner.hotkey,
                    uid=result.miner.uid,
                    current_revision=result.miner.revision or 'unknown',
                    model_hash=result.miner.model or 'unknown',
                    model_name=result.miner.model or 'unknown',
                    last_commit_block=result.miner.block
                )
            
            self.stats['total_migrated'] += 1
            return True
            
        except Exception as e:
            print(f"Error migrating result: {e}")
            self.stats['total_errors'] += 1
            return False
    
    async def migrate_batch(self, results: List[Result], batch_size: int = 100):
        """Migrate a batch of results.
        
        Args:
            results: List of results to migrate
            batch_size: Number of results to process concurrently
        """
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            
            tasks = [self.migrate_result(r) for r in batch]
            await asyncio.gather(*tasks)
            
            self.stats['total_processed'] += len(batch)
            
            # Print progress
            if self.stats['total_processed'] % 1000 == 0:
                elapsed = time.time() - self.stats['start_time']
                rate = self.stats['total_processed'] / elapsed
                print(f"Processed {self.stats['total_processed']} results "
                      f"({rate:.1f} results/sec)")
    
    async def migrate_from_r2(
        self,
        tail_blocks: int = 100000,
        batch_size: int = 100,
        max_results: int = None
    ):
        """Migrate data from R2 storage.
        
        Args:
            tail_blocks: Number of blocks to look back
            batch_size: Batch size for concurrent processing
            max_results: Maximum number of results to migrate (None = all)
        """
        print(f"Starting migration from R2 (tail={tail_blocks} blocks)")
        
        results_buffer = []
        count = 0
        
        async for result in dataset(tail=tail_blocks, compact=False):
            results_buffer.append(result)
            count += 1
            
            # Process batch
            if len(results_buffer) >= batch_size:
                await self.migrate_batch(results_buffer, batch_size)
                results_buffer = []
            
            # Check max limit
            if max_results and count >= max_results:
                break
        
        # Process remaining results
        if results_buffer:
            await self.migrate_batch(results_buffer, batch_size)
        
        self.print_summary()
    
    def print_summary(self):
        """Print migration summary."""
        elapsed = time.time() - self.stats['start_time']
        
        print("\n" + "="*60)
        print("Migration Summary")
        print("="*60)
        print(f"Total Processed:  {self.stats['total_processed']}")
        print(f"Total Migrated:   {self.stats['total_migrated']}")
        print(f"Total Skipped:    {self.stats['total_skipped']}")
        print(f"Total Errors:     {self.stats['total_errors']}")
        print(f"Miners Updated:   {len(self.stats['miners_updated'])}")
        print(f"Elapsed Time:     {elapsed:.1f}s")
        print(f"Migration Rate:   {self.stats['total_processed']/elapsed:.1f} results/sec")
        print("="*60)


async def run_migration(tail_blocks: int = 100000, max_results: int = None):
    """Run the migration process.
    
    Args:
        tail_blocks: Number of blocks to look back
        max_results: Maximum number of results to migrate
    """
    # Initialize DynamoDB client
    await init_client()
    
    try:
        migration = R2ToDynamoMigration()
        await migration.migrate_from_r2(
            tail_blocks=tail_blocks,
            max_results=max_results
        )
    finally:
        # Close DynamoDB client
        await close_client()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate data from R2 to DynamoDB")
    parser.add_argument(
        "--tail",
        type=int,
        default=100000,
        help="Number of blocks to look back (default: 100000)"
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
        max_results=args.max_results
    ))