"""
Database CLI tool for management and testing

Provides commands for initializing, testing, and managing DynamoDB tables.
"""

import asyncio
import sys
from typing import Optional

from affine.database import init_client, close_client, init_tables
from affine.database.tables import list_tables, reset_tables, delete_table
from affine.database.dao import (
    SampleResultsDAO,
    TaskQueueDAO,
    ExecutionLogsDAO,
    ScoresDAO,
    SystemConfigDAO,
    MinerMetadataDAO,
    DataRetentionDAO,
)


async def cmd_init():
    """Initialize all DynamoDB tables."""
    print("Initializing DynamoDB tables...")
    await init_client()
    
    try:
        await init_tables()
        print("✓ Tables initialized successfully")
    finally:
        await close_client()


async def cmd_list():
    """List all tables."""
    await init_client()
    
    try:
        tables = await list_tables()
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
    finally:
        await close_client()


async def cmd_reset():
    """Reset all tables (delete and recreate)."""
    confirm = input("WARNING: This will delete all data. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        await reset_tables()
        print("✓ Tables reset successfully")
    finally:
        await close_client()


async def cmd_test_basic():
    """Run basic CRUD tests on all DAOs."""
    print("Running basic DAO tests...")
    await init_client()
    
    try:
        # Test SampleResultsDAO
        print("\n1. Testing SampleResultsDAO...")
        sample_dao = SampleResultsDAO()
        
        sample = await sample_dao.save_sample(
            miner_hotkey="test_hotkey_123",
            model_revision="v1.0.0",
            model="test-org/test-model",
            uid=42,
            env="L3",
            subset="test",
            dataset_index=0,
            task_id="task_001",
            score=0.95,
            success=True,
            latency_ms=150,
            conversation={"messages": [{"role": "user", "content": "test"}]},
            validator_hotkey="validator_123",
            block_number=1000000,
            signature="sig_123"
        )
        print(f"  ✓ Saved sample: {sample['sample_id']}")
        
        # Retrieve samples
        samples = await sample_dao.get_samples_by_miner(
            "test_hotkey_123",
            "v1.0.0",
            limit=10
        )
        print(f"  ✓ Retrieved {len(samples)} samples")
        
        # Test TaskQueueDAO
        print("\n2. Testing TaskQueueDAO...")
        task_dao = TaskQueueDAO()
        
        task = await task_dao.create_task(
            miner_hotkey="test_hotkey_123",
            model_revision="v1.0.0",
            model="test-org/test-model",
            env="L3",
            validator_hotkey="validator_123"
        )
        print(f"  ✓ Created task: {task['task_id']}")
        
        # Start task
        await task_dao.start_task("test_hotkey_123", "v1.0.0", task['task_id'])
        print(f"  ✓ Started task")
        
        # Complete task
        await task_dao.complete_task("test_hotkey_123", "v1.0.0", task['task_id'])
        print(f"  ✓ Completed task")
        
        # Test ExecutionLogsDAO
        print("\n3. Testing ExecutionLogsDAO...")
        log_dao = ExecutionLogsDAO()
        
        log = await log_dao.log_execution(
            miner_hotkey="test_hotkey_123",
            uid=42,
            task_id="task_001",
            status="success",
            env="L3",
            execution_time_ms=150
        )
        print(f"  ✓ Created log: {log['log_id']}")
        
        # Get recent logs
        logs = await log_dao.get_recent_logs("test_hotkey_123", limit=10)
        print(f"  ✓ Retrieved {len(logs)} logs")
        
        # Test ScoresDAO
        print("\n4. Testing ScoresDAO...")
        score_dao = ScoresDAO()
        
        score = await score_dao.save_score(
            block_number=1000000,
            miner_hotkey="test_hotkey_123",
            uid=42,
            model_revision="v1.0.0",
            overall_score=0.95,
            confidence_interval_lower=0.90,
            confidence_interval_upper=0.98,
            average_score=0.94,
            scores_by_layer={"L3": 0.95, "L4": 0.93},
            scores_by_env={"env1": 0.96, "env2": 0.94},
            total_samples=100,
            is_eligible=True,
            meets_criteria=True
        )
        print(f"  ✓ Saved score at block {score['block_number']}")
        
        # Get latest scores
        latest = await score_dao.get_latest_scores()
        print(f"  ✓ Retrieved latest scores: block {latest.get('block_number')}")
        
        # Test SystemConfigDAO
        print("\n5. Testing SystemConfigDAO...")
        config_dao = SystemConfigDAO()
        
        await config_dao.set_param(
            param_name="test_param",
            param_value=123,
            param_type="int",
            description="Test parameter"
        )
        print(f"  ✓ Set config parameter")
        
        value = await config_dao.get_param_value("test_param")
        print(f"  ✓ Retrieved config value: {value}")
        
        # Test MinerMetadataDAO
        print("\n6. Testing MinerMetadataDAO...")
        miner_dao = MinerMetadataDAO()
        
        metadata = await miner_dao.save_metadata(
            miner_hotkey="test_hotkey_123",
            uid=42,
            current_revision="v1.0.0",
            model="test-org/test-model",
            model_name="test_model",
            last_commit_block=1000000
        )
        print(f"  ✓ Saved miner metadata")
        
        # Update statistics
        await miner_dao.update_statistics("test_hotkey_123", success=True)
        print(f"  ✓ Updated miner statistics")
        
        # Test DataRetentionDAO
        print("\n7. Testing DataRetentionDAO...")
        retention_dao = DataRetentionDAO()
        
        policy = await retention_dao.set_protected(
            miner_hotkey="test_hotkey_123",
            reason="Test protection"
        )
        print(f"  ✓ Set retention policy")
        
        is_protected = await retention_dao.is_protected("test_hotkey_123")
        print(f"  ✓ Checked protection status: {is_protected}")
        
        print("\n✓ All basic tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await close_client()


async def cmd_migrate(tail: int, max_results: Optional[int]):
    """Run migration from R2 to DynamoDB."""
    from affine.database.migrate import run_migration
    
    print(f"Starting migration (tail={tail}, max_results={max_results or 'all'})")
    await run_migration(tail_blocks=tail, max_results=max_results)


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DynamoDB database management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    subparsers.add_parser("init", help="Initialize all tables")
    
    # List command
    subparsers.add_parser("list", help="List all tables")
    
    # Reset command
    subparsers.add_parser("reset", help="Reset all tables (delete and recreate)")
    
    # Test command
    subparsers.add_parser("test", help="Run basic CRUD tests")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate data from R2")
    migrate_parser.add_argument(
        "--tail",
        type=int,
        default=100000,
        help="Number of blocks to look back"
    )
    migrate_parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum results to migrate"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == "init":
        asyncio.run(cmd_init())
    elif args.command == "list":
        asyncio.run(cmd_list())
    elif args.command == "reset":
        asyncio.run(cmd_reset())
    elif args.command == "test":
        asyncio.run(cmd_test_basic())
    elif args.command == "migrate":
        asyncio.run(cmd_migrate(args.tail, args.max_results))
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()