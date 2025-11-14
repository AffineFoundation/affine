"""
System Config DAO

Manages dynamic configuration parameters.
"""

from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class SystemConfigDAO(BaseDAO):
    """DAO for system_config table.
    
    Stores dynamic configuration parameters.
    PK: CONFIG
    SK: PARAM#{param_name}
    """
    
    def __init__(self):
        self.table_name = get_table_name("system_config")
        super().__init__()
    
    def _make_pk(self) -> str:
        """Generate partition key."""
        return "CONFIG"
    
    def _make_sk(self, param_name: str) -> str:
        """Generate sort key."""
        return f"PARAM#{param_name}"
    
    async def set_param(
        self,
        param_name: str,
        param_value: Any,
        param_type: str,
        description: str = "",
        updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Set a configuration parameter.
        
        Args:
            param_name: Parameter name
            param_value: Parameter value
            param_type: Parameter type (str/int/float/bool/dict/list)
            description: Parameter description
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        import time
        
        # Get existing config to increment version
        existing = await self.get_param(param_name)
        version = (existing.get('version', 0) + 1) if existing else 1
        
        item = {
            'pk': self._make_pk(),
            'sk': self._make_sk(param_name),
            'param_name': param_name,
            'param_value': param_value,
            'param_type': param_type,
            'description': description,
            'updated_at': int(time.time()),
            'updated_by': updated_by,
            'version': version,
        }
        
        return await self.put(item)
    
    async def get_param(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get a configuration parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Config item if found, None otherwise
        """
        pk = self._make_pk()
        sk = self._make_sk(param_name)
        
        return await self.get(pk, sk)
    
    async def get_param_value(self, param_name: str, default: Any = None) -> Any:
        """Get parameter value directly.
        
        Args:
            param_name: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value or default
        """
        param = await self.get_param(param_name)
        return param.get('param_value', default) if param else default
    
    async def get_all_params(self) -> Dict[str, Any]:
        """Get all configuration parameters.
        
        Returns:
            Dictionary mapping param names to values
        """
        pk = self._make_pk()
        
        items = await self.query(pk=pk)
        
        return {item['param_name']: item['param_value'] for item in items}
    
    async def delete_param(self, param_name: str) -> bool:
        """Delete a configuration parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            True if deleted successfully
        """
        pk = self._make_pk()
        sk = self._make_sk(param_name)
        
        return await self.delete(pk, sk)
    
    async def list_all_configs(self) -> List[Dict[str, Any]]:
        """List all configuration parameters with metadata.
        
        Returns:
            List of config items with full details
        """
        pk = self._make_pk()
        
        return await self.query(pk=pk)