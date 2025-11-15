"""
DynamoDB GSI Management Tool

Provides utilities to add/remove Global Secondary Indexes online without rebuilding tables.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from affine.database.client import get_client, get_table_prefix


class GSIManager:
    """Manager for DynamoDB Global Secondary Indexes."""
    
    def __init__(self):
        self.client = None
        self.table_prefix = get_table_prefix()
    
    async def _get_client(self):
        """Get DynamoDB client."""
        if not self.client:
            self.client = get_client()
        return self.client
    
    def _get_table_name(self, base_name: str) -> str:
        """Get full table name with prefix."""
        return f"{self.table_prefix}_{base_name}"
    
    async def add_gsi(
        self,
        table_name: str,
        index_name: str,
        hash_key: str,
        hash_key_type: str,
        range_key: Optional[str] = None,
        range_key_type: Optional[str] = None,
        projection_type: str = "ALL"
    ) -> bool:
        """
        Add a Global Secondary Index to an existing table.
        
        Args:
            table_name: Base table name (without prefix)
            index_name: Name of the GSI
            hash_key: Partition key attribute name
            hash_key_type: Partition key type (S/N/B)
            range_key: Sort key attribute name (optional)
            range_key_type: Sort key type (S/N/B, optional)
            projection_type: Projection type (ALL/KEYS_ONLY/INCLUDE)
            
        Returns:
            True if GSI was added successfully
        """
        client = await self._get_client()
        full_table_name = self._get_table_name(table_name)
        
        # Build key schema
        key_schema = [{"AttributeName": hash_key, "KeyType": "HASH"}]
        if range_key:
            key_schema.append({"AttributeName": range_key, "KeyType": "RANGE"})
        
        # Build attribute definitions
        attribute_definitions = [{"AttributeName": hash_key, "AttributeType": hash_key_type}]
        if range_key:
            attribute_definitions.append({"AttributeName": range_key, "AttributeType": range_key_type})
        
        try:
            print(f"Adding GSI '{index_name}' to table '{full_table_name}'...")
            
            response = await client.update_table(
                TableName=full_table_name,
                AttributeDefinitions=attribute_definitions,
                GlobalSecondaryIndexUpdates=[
                    {
                        "Create": {
                            "IndexName": index_name,
                            "KeySchema": key_schema,
                            "Projection": {"ProjectionType": projection_type}
                        }
                    }
                ]
            )
            
            print(f"✓ GSI '{index_name}' is being created (status: CREATING)")
            print(f"  This will take a few minutes depending on table size.")
            print(f"  You can check status with: python -m affine.database.manage_gsi check-gsi {table_name} {index_name}")
            
            return True
            
        except Exception as e:
            if "ResourceInUseException" in str(e):
                print(f"✗ Table is currently being updated. Please wait and try again.")
            elif "ValidationException" in str(e):
                print(f"✗ GSI '{index_name}' may already exist or invalid parameters: {e}")
            else:
                print(f"✗ Failed to add GSI: {e}")
            return False
    
    async def remove_gsi(self, table_name: str, index_name: str) -> bool:
        """
        Remove a Global Secondary Index from a table.
        
        Args:
            table_name: Base table name (without prefix)
            index_name: Name of the GSI to remove
            
        Returns:
            True if GSI was removed successfully
        """
        client = await self._get_client()
        full_table_name = self._get_table_name(table_name)
        
        try:
            print(f"Removing GSI '{index_name}' from table '{full_table_name}'...")
            
            await client.update_table(
                TableName=full_table_name,
                GlobalSecondaryIndexUpdates=[
                    {
                        "Delete": {
                            "IndexName": index_name
                        }
                    }
                ]
            )
            
            print(f"✓ GSI '{index_name}' is being deleted")
            return True
            
        except Exception as e:
            print(f"✗ Failed to remove GSI: {e}")
            return False
    
    async def list_gsi(self, table_name: str) -> List[Dict[str, Any]]:
        """
        List all Global Secondary Indexes for a table.
        
        Args:
            table_name: Base table name (without prefix)
            
        Returns:
            List of GSI definitions
        """
        client = await self._get_client()
        full_table_name = self._get_table_name(table_name)
        
        try:
            response = await client.describe_table(TableName=full_table_name)
            table_info = response.get("Table", {})
            gsi_list = table_info.get("GlobalSecondaryIndexes", [])
            
            print(f"\nGlobal Secondary Indexes for '{full_table_name}':")
            print(f"{'='*80}")
            
            if not gsi_list:
                print("  (No GSI found)")
            else:
                for gsi in gsi_list:
                    index_name = gsi["IndexName"]
                    status = gsi.get("IndexStatus", "UNKNOWN")
                    key_schema = gsi["KeySchema"]
                    
                    hash_key = next((k["AttributeName"] for k in key_schema if k["KeyType"] == "HASH"), None)
                    range_key = next((k["AttributeName"] for k in key_schema if k["KeyType"] == "RANGE"), None)
                    
                    print(f"\n  Index: {index_name}")
                    print(f"    Status: {status}")
                    print(f"    Hash Key: {hash_key}")
                    if range_key:
                        print(f"    Range Key: {range_key}")
                    print(f"    Projection: {gsi['Projection']['ProjectionType']}")
            
            print(f"\n{'='*80}\n")
            return gsi_list
            
        except Exception as e:
            print(f"✗ Failed to list GSI: {e}")
            return []
    
    async def check_gsi_status(self, table_name: str, index_name: str) -> Optional[str]:
        """
        Check the status of a specific GSI.
        
        Args:
            table_name: Base table name (without prefix)
            index_name: Name of the GSI
            
        Returns:
            Status string (CREATING/ACTIVE/DELETING) or None if not found
        """
        client = await self._get_client()
        full_table_name = self._get_table_name(table_name)
        
        try:
            response = await client.describe_table(TableName=full_table_name)
            table_info = response.get("Table", {})
            gsi_list = table_info.get("GlobalSecondaryIndexes", [])
            
            for gsi in gsi_list:
                if gsi["IndexName"] == index_name:
                    status = gsi.get("IndexStatus", "UNKNOWN")
                    print(f"GSI '{index_name}' status: {status}")
                    return status
            
            print(f"✗ GSI '{index_name}' not found")
            return None
            
        except Exception as e:
            print(f"✗ Failed to check GSI status: {e}")
            return None


async def main():
    """CLI interface for GSI management."""
    import sys
    
    if len(sys.argv) < 2:
        print("""
DynamoDB GSI Management Tool

Usage:
  python -m affine.database.manage_gsi <command> [args]

Commands:
  add-gsi <table> <index_name> <hash_key> <hash_type> [range_key] [range_type]
    Add a new GSI to a table
    Example: python -m affine.database.manage_gsi add-gsi sample_results block-timestamp-index block_number N timestamp N

  remove-gsi <table> <index_name>
    Remove a GSI from a table
    Example: python -m affine.database.manage_gsi remove-gsi sample_results old-index

  list-gsi <table>
    List all GSI for a table
    Example: python -m affine.database.manage_gsi list-gsi sample_results

  check-gsi <table> <index_name>
    Check status of a specific GSI
    Example: python -m affine.database.manage_gsi check-gsi sample_results block-timestamp-index
""")
        return
    
    command = sys.argv[1]
    manager = GSIManager()
    
    if command == "add-gsi":
        if len(sys.argv) < 6:
            print("Error: add-gsi requires at least: <table> <index_name> <hash_key> <hash_type>")
            return
        
        table_name = sys.argv[2]
        index_name = sys.argv[3]
        hash_key = sys.argv[4]
        hash_key_type = sys.argv[5]
        range_key = sys.argv[6] if len(sys.argv) > 6 else None
        range_key_type = sys.argv[7] if len(sys.argv) > 7 else None
        
        await manager.add_gsi(
            table_name=table_name,
            index_name=index_name,
            hash_key=hash_key,
            hash_key_type=hash_key_type,
            range_key=range_key,
            range_key_type=range_key_type
        )
    
    elif command == "remove-gsi":
        if len(sys.argv) < 4:
            print("Error: remove-gsi requires: <table> <index_name>")
            return
        
        table_name = sys.argv[2]
        index_name = sys.argv[3]
        await manager.remove_gsi(table_name, index_name)
    
    elif command == "list-gsi":
        if len(sys.argv) < 3:
            print("Error: list-gsi requires: <table>")
            return
        
        table_name = sys.argv[2]
        await manager.list_gsi(table_name)
    
    elif command == "check-gsi":
        if len(sys.argv) < 4:
            print("Error: check-gsi requires: <table> <index_name>")
            return
        
        table_name = sys.argv[2]
        index_name = sys.argv[3]
        await manager.check_gsi_status(table_name, index_name)
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    asyncio.run(main())