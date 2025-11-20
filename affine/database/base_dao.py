"""
Base DAO providing common DynamoDB operations

Implements the Data Access Object pattern with standard CRUD operations.
"""

import time
import gzip
from typing import Dict, Any, List, Optional
from decimal import Decimal
from affine.database.client import get_client


class BaseDAO:
    """Base class for DynamoDB data access objects.
    
    Provides common operations like put, get, query, scan, delete.
    Subclasses define table_name and implement specific business logic.
    """
    
    table_name: str = None  # Override in subclass
    
    def __init__(self):
        if self.table_name is None:
            raise ValueError("table_name must be defined in subclass")
    
    async def put(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Put an item into the table.
        
        Args:
            item: Item to insert/update
            
        Returns:
            The item that was inserted
        """
        client = get_client()
        
        # Convert floats to Decimal for DynamoDB
        item = self._serialize(item)
        
        await client.put_item(
            TableName=self.table_name,
            Item=item
        )
        
        return item
    
    async def get(self, pk: str, sk: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get an item by primary key.
        
        Args:
            pk: Partition key value
            sk: Sort key value (optional, for tables with composite keys)
            
        Returns:
            Item if found, None otherwise
        """
        client = get_client()
        
        key = {'pk': {'S': pk}}
        if sk is not None:
            key['sk'] = {'S': sk}
        
        response = await client.get_item(
            TableName=self.table_name,
            Key=key
        )
        
        item = response.get('Item')
        return self._deserialize(item) if item else None
    
    async def query(
        self,
        pk: str,
        sk_prefix: Optional[str] = None,
        index_name: Optional[str] = None,
        limit: Optional[int] = None,
        reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """Query items by partition key and optional sort key prefix.
        
        Args:
            pk: Partition key value
            sk_prefix: Optional sort key prefix for filtering
            index_name: Optional GSI name
            limit: Maximum number of items to return
            reverse: If True, return items in descending order
            
        Returns:
            List of matching items
        """
        client = get_client()
        
        # Build key condition
        if sk_prefix:
            key_condition = 'pk = :pk AND begins_with(sk, :sk)'
            expression_values = {
                ':pk': {'S': pk},
                ':sk': {'S': sk_prefix}
            }
        else:
            key_condition = 'pk = :pk'
            expression_values = {':pk': {'S': pk}}
        
        params = {
            'TableName': self.table_name,
            'KeyConditionExpression': key_condition,
            'ExpressionAttributeValues': expression_values,
            'ScanIndexForward': not reverse
        }
        
        if index_name:
            params['IndexName'] = index_name
        
        if limit:
            params['Limit'] = limit
        
        items = []
        
        while True:
            response = await client.query(**params)
            items.extend([self._deserialize(item) for item in response.get('Items', [])])
            
            # Check if we have more results
            last_key = response.get('LastEvaluatedKey')
            if not last_key or (limit and len(items) >= limit):
                break
            
            params['ExclusiveStartKey'] = last_key
        
        return items[:limit] if limit else items
    
    async def delete(self, pk: str, sk: Optional[str] = None) -> bool:
        """Delete an item by primary key.
        
        Args:
            pk: Partition key value
            sk: Sort key value (optional, for tables with composite keys)
            
        Returns:
            True if item was deleted, False if not found
        """
        client = get_client()
        
        try:
            key = {'pk': {'S': pk}}
            if sk is not None:
                key['sk'] = {'S': sk}
            
            await client.delete_item(
                TableName=self.table_name,
                Key=key
            )
            return True
        except Exception:
            return False
    
    async def batch_write(self, items: List[Dict[str, Any]]):
        """Batch write items to the table.
        
        Args:
            items: List of items to write (max 25 per batch)
        """
        client = get_client()
        
        # Process in chunks of 25 (DynamoDB limit)
        for i in range(0, len(items), 25):
            batch = items[i:i + 25]
            
            request_items = {
                self.table_name: [
                    {'PutRequest': {'Item': self._serialize(item)}}
                    for item in batch
                ]
            }
            
            await client.batch_write_item(RequestItems=request_items)
    
    def _serialize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Python types to DynamoDB format.
        
        Args:
            item: Python dict with standard types
            
        Returns:
            DynamoDB-formatted dict
        """
        def convert_value(value):
            if value is None:
                return {'NULL': True}
            elif isinstance(value, bool):
                return {'BOOL': value}
            elif isinstance(value, (int, Decimal)):
                return {'N': str(value)}
            elif isinstance(value, float):
                return {'N': str(Decimal(str(value)))}
            elif isinstance(value, str):
                return {'S': value}
            elif isinstance(value, bytes):
                return {'B': value}
            elif isinstance(value, list):
                return {'L': [convert_value(v) for v in value]}
            elif isinstance(value, dict):
                return {'M': {k: convert_value(v) for k, v in value.items()}}
            else:
                return {'S': str(value)}
        
        return {k: convert_value(v) for k, v in item.items()}
    
    def _deserialize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DynamoDB format to Python types.
        
        Args:
            item: DynamoDB-formatted dict
            
        Returns:
            Python dict with standard types
        """
        def convert_value(value_dict):
            if 'NULL' in value_dict:
                return None
            elif 'BOOL' in value_dict:
                return value_dict['BOOL']
            elif 'N' in value_dict:
                num_str = value_dict['N']
                return int(num_str) if '.' not in num_str else float(num_str)
            elif 'S' in value_dict:
                return value_dict['S']
            elif 'B' in value_dict:
                return value_dict['B']
            elif 'L' in value_dict:
                return [convert_value(v) for v in value_dict['L']]
            elif 'M' in value_dict:
                return {k: convert_value(v) for k, v in value_dict['M'].items()}
            else:
                return None
        
        return {k: convert_value(v) for k, v in item.items()}
    
    @staticmethod
    def compress_data(data: str) -> bytes:
        """Compress string data using gzip.
        
        Args:
            data: String to compress
            
        Returns:
            Compressed bytes
        """
        return gzip.compress(data.encode('utf-8'))
    
    @staticmethod
    def decompress_data(data: bytes) -> str:
        """Decompress gzip data.
        
        Args:
            data: Compressed bytes
            
        Returns:
            Decompressed string
        """
        return gzip.decompress(data).decode('utf-8')
    
    @staticmethod
    def get_ttl(days: int) -> int:
        """Get TTL timestamp for given number of days from now.
        
        Args:
            days: Number of days until expiration
            
        Returns:
            Unix timestamp
        """
        return int(time.time()) + (days * 86400)