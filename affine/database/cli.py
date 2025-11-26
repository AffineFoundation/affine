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


def main():
    """Main CLI entry point."""
    db()


if __name__ == "__main__":
    main()