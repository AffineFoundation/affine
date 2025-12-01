"""
Database CLI tool for management and testing

Provides commands for initializing, testing, and managing DynamoDB tables.
"""

import asyncio
import sys
from typing import Optional
import os
import click

from affine.database import init_client, close_client, init_tables
from affine.database.tables import list_tables, reset_tables, delete_table
from affine.database.dao import (
    SampleResultsDAO,
    TaskPoolDAO,
    ExecutionLogsDAO,
    ScoresDAO,
    SystemConfigDAO,
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


async def cmd_reset_table(table_name: str):
    """Reset a single table (delete and recreate)."""
    from affine.database.schema import get_table_name
    
    # Get full table name with environment prefix
    full_table_name = get_table_name(table_name)
    
    confirm = input(f"WARNING: This will delete all data in '{full_table_name}'. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        print(f"Deleting table '{full_table_name}'...")
        await delete_table(full_table_name)
        
        print(f"Recreating table '{full_table_name}'...")
        await init_tables()
        
        print(f"✓ Table '{full_table_name}' reset successfully")
    except Exception as e:
        print(f"✗ Failed to reset table: {e}")
        sys.exit(1)
    finally:
        await close_client()


async def cmd_migrate(tail: int, max_results: Optional[int]):
    """Run migration from R2 to DynamoDB."""
    from affine.database.migrate import run_migration
    
    print(f"Starting migration (tail={tail}, max_results={max_results or 'all'})")
    await run_migration(tail_blocks=tail, max_results=max_results)


async def cmd_load_config(json_file: str):
    """Load system configuration from JSON file."""
    import json
    import os
    
    print(f"Loading configuration from {json_file}...")
    
    # Check file exists
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    
    # Load JSON
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)
    
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        # Load validator burn percentage if present
        if 'validator_burn_percentage' in config:
            burn_percentage = float(config['validator_burn_percentage'])
            
            await config_dao.set_param(
                param_name='validator_burn_percentage',
                param_value=burn_percentage,
                param_type='float',
                description='Percentage of weight to burn (allocate to UID 0)',
                updated_by='cli_load_config'
            )
            
            print(f"✓ Loaded validator burn percentage: {burn_percentage:.1%}")
        
        # Load new unified environments structure
        if 'environments' in config:
            environments = config['environments']
            
            # Save to database
            await config_dao.set_param(
                param_name='environments',
                param_value=environments,
                param_type='dict',
                description='Environment configurations (sampling, scoring, ranges)',
                updated_by='cli_load_config'
            )
            
            print(f"✓ Loaded configuration for {len(environments)} environments:")
            
            for env_name, env_config in environments.items():
                enabled_sampling = env_config.get('enabled_for_sampling', False)
                enabled_scoring = env_config.get('enabled_for_scoring', False)
                sampling_range = env_config.get('sampling_range', [0, 0])
                scoring_range = env_config.get('scoring_range', [0, 0])
                
                status = []
                if enabled_sampling:
                    status.append(f"sampling: {sampling_range}")
                if enabled_scoring:
                    status.append(f"scoring: {scoring_range}")
                
                status_str = ", ".join(status) if status else "disabled"
                print(f"  {env_name}: {status_str}")
        
        print("\n✓ Configuration loaded successfully!")
        
    finally:
        await close_client()


async def cmd_blacklist_list():
    """List all blacklisted hotkeys."""
    print("Fetching blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        blacklist = await config_dao.get_blacklist()
        
        if not blacklist:
            print("Blacklist is empty")
        else:
            print(f"Blacklist contains {len(blacklist)} hotkey(s):")
            for i, hotkey in enumerate(blacklist, 1):
                print(f"  {i}. {hotkey}")
    
    finally:
        await close_client()


async def cmd_blacklist_add(hotkeys: list):
    """Add hotkeys to blacklist."""
    print(f"Adding {len(hotkeys)} hotkey(s) to blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        # Get current blacklist
        current = await config_dao.get_blacklist()
        print(f"Current blacklist size: {len(current)}")
        
        # Add new hotkeys
        result = await config_dao.add_to_blacklist(
            hotkeys=hotkeys,
            updated_by='cli_blacklist_add'
        )
        
        new_blacklist = result.get('param_value', [])
        print(f"✓ Updated blacklist size: {len(new_blacklist)}")
        print(f"  Added: {len(new_blacklist) - len(current)} new hotkey(s)")
        
    finally:
        await close_client()


async def cmd_blacklist_remove(hotkeys: list):
    """Remove hotkeys from blacklist."""
    print(f"Removing {len(hotkeys)} hotkey(s) from blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        # Get current blacklist
        current = await config_dao.get_blacklist()
        print(f"Current blacklist size: {len(current)}")
        
        # Remove hotkeys
        result = await config_dao.remove_from_blacklist(
            hotkeys=hotkeys,
            updated_by='cli_blacklist_remove'
        )
        
        new_blacklist = result.get('param_value', [])
        print(f"✓ Updated blacklist size: {len(new_blacklist)}")
        print(f"  Removed: {len(current) - len(new_blacklist)} hotkey(s)")
        
    finally:
        await close_client()


async def cmd_blacklist_clear():
    """Clear all hotkeys from blacklist."""
    confirm = input("WARNING: This will clear the entire blacklist. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    print("Clearing blacklist...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        result = await config_dao.set_blacklist(
            hotkeys=[],
            updated_by='cli_blacklist_clear'
        )
        
        print("✓ Blacklist cleared successfully")
        
    finally:
        await close_client()


async def cmd_set_burn_percentage(burn_percentage: float):
    """Set validator burn percentage."""
    if burn_percentage < 0 or burn_percentage > 1:
        print(f"Error: Burn percentage must be between 0 and 1 (got {burn_percentage})")
        sys.exit(1)
    
    print(f"Setting burn percentage to {burn_percentage:.1%}...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        
        result = await config_dao.set_param(
            param_name='validator_burn_percentage',
            param_value=burn_percentage,
            param_type='float',
            description='Percentage of weight to burn (allocate to UID 0)',
            updated_by='cli_set_burn_percentage'
        )
        
        print(f"✓ Burn percentage set to {burn_percentage:.1%}")
        
    finally:
        await close_client()


async def cmd_get_burn_percentage():
    """Get current validator burn percentage."""
    print("Fetching burn percentage...")
    await init_client()
    
    try:
        config_dao = SystemConfigDAO()
        config = await config_dao.get_param('validator_burn_percentage')
        
        if not config:
            print("Burn percentage not set (default: 0.0)")
        else:
            burn_percentage = config.get('param_value', 0.0)
            print(f"Current burn percentage: {burn_percentage:.1%}")
            print(f"Last updated: {config.get('updated_at', 'unknown')}")
            print(f"Updated by: {config.get('updated_by', 'unknown')}")
    
    finally:
        await close_client()


async def cmd_delete_samples_by_range(
    hotkey: str,
    revision: str,
    env: str,
    start_task_id: int,
    end_task_id: int
):
    """Delete samples within a task_id range."""
    print(f"Deleting samples for hotkey={hotkey[:12]}..., env={env}, task_id range=[{start_task_id}, {end_task_id})...")
    
    confirm = input(f"WARNING: This will delete samples in range [{start_task_id}, {end_task_id}). Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        sample_dao = SampleResultsDAO()
        deleted_count = await sample_dao.delete_samples_by_task_range(
            miner_hotkey=hotkey,
            model_revision=revision,
            env=env,
            start_task_id=start_task_id,
            end_task_id=end_task_id
        )
        
        print(f"✓ Deleted {deleted_count} samples")
    
    except Exception as e:
        print(f"✗ Failed to delete samples: {e}")
        sys.exit(1)
    finally:
        await close_client()


async def cmd_delete_samples_empty_conversation():
    """Delete all samples with empty conversation across the entire database."""
    print("Scanning entire sample database for invalid samples (empty conversation)...")
    
    confirm = input("WARNING: This will scan and delete ALL samples with empty conversation in the database. Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Aborted")
        return
    
    await init_client()
    
    try:
        sample_dao = SampleResultsDAO()
        deleted_count = await sample_dao.delete_all_samples_with_empty_conversation()
        
        print(f"\n✓ Scan complete. Deleted {deleted_count} samples with empty conversation")
    
    except Exception as e:
        print(f"\n✗ Failed to delete samples: {e}")
        sys.exit(1)
    finally:
        await close_client()


@click.group()
def db():
    """Database management commands."""
    pass


@db.command()
def init():
    """Initialize all DynamoDB tables."""
    asyncio.run(cmd_init())


@db.command("list")
def list_cmd():
    """List all tables."""
    asyncio.run(cmd_list())


@db.command()
def reset():
    """Reset all tables (delete and recreate)."""
    asyncio.run(cmd_reset())


@db.command("reset-table")
@click.option("--table", required=True, help="Table name to reset (e.g., task_queue, sample_results)")
def reset_table(table):
    """Reset a single table (delete and recreate)."""
    asyncio.run(cmd_reset_table(table))


@db.command()
@click.option("--tail", type=int, default=100000, help="Number of blocks to look back")
@click.option("--max-results", type=int, default=None, help="Maximum results to migrate")
def migrate(tail, max_results):
    """Migrate data from R2."""
    asyncio.run(cmd_migrate(tail, max_results))


@db.command("load-config")
@click.option(
    "--json-file",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_config.json"),
    help="Path to JSON configuration file"
)
def load_config(json_file):
    """Load system configuration from JSON file."""
    asyncio.run(cmd_load_config(json_file))


@db.group()
def blacklist():
    """Manage miner blacklist."""
    pass


@blacklist.command("list")
def blacklist_list():
    """List all blacklisted hotkeys."""
    asyncio.run(cmd_blacklist_list())


@blacklist.command()
@click.argument("hotkeys", nargs=-1, required=True)
def add(hotkeys):
    """Add hotkeys to blacklist."""
    asyncio.run(cmd_blacklist_add(list(hotkeys)))


@blacklist.command()
@click.argument("hotkeys", nargs=-1, required=True)
def remove(hotkeys):
    """Remove hotkeys from blacklist."""
    asyncio.run(cmd_blacklist_remove(list(hotkeys)))


@blacklist.command()
def clear():
    """Clear all hotkeys from blacklist."""
    asyncio.run(cmd_blacklist_clear())


@db.command("set-burn")
@click.argument("percentage", type=float)
def set_burn(percentage):
    """Set validator burn percentage (0.0 to 1.0)."""
    asyncio.run(cmd_set_burn_percentage(percentage))


@db.command("get-burn")
def get_burn():
    """Get current validator burn percentage."""
    asyncio.run(cmd_get_burn_percentage())


@db.command("delete-samples-by-range")
@click.option("--hotkey", required=True, help="Miner's hotkey")
@click.option("--revision", required=True, help="Model revision hash")
@click.option("--env", required=True, help="Environment name (e.g., agentgym:alfworld)")
@click.option("--start-task-id", required=True, type=int, help="Start task_id (inclusive)")
@click.option("--end-task-id", required=True, type=int, help="End task_id (exclusive)")
def delete_samples_by_range(hotkey, revision, env, start_task_id, end_task_id):
    """Delete samples within a task_id range for a specific miner and environment.
    
    Example:
        af db delete-samples-by-range --hotkey 5C5... --revision abc123 --env agentgym:alfworld --start-task-id 0 --end-task-id 100
    """
    asyncio.run(cmd_delete_samples_by_range(hotkey, revision, env, start_task_id, end_task_id))


@db.command("delete-samples-empty-conversation")
def delete_samples_empty_conversation():
    """Delete all samples with empty conversation across the entire database.
    
    This command will scan the entire sample_results table and delete any samples
    where the conversation field is empty or null. Progress will be logged during execution.
    
    Example:
        af db delete-samples-empty-conversation
    """
    asyncio.run(cmd_delete_samples_empty_conversation())


def main():
    """Main CLI entry point."""
    db()


if __name__ == "__main__":
    main()