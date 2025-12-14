"""
System Config DAO

Manages dynamic configuration parameters.
"""

from typing import Dict, Any, List, Optional, Set
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


def validate_ranges(ranges: List[List[int]]) -> None:
    """Validate multi-range format.
    
    Args:
        ranges: List of [start, end] pairs
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(ranges, list):
        raise ValueError("Ranges must be a list")
    
    if len(ranges) == 0:
        raise ValueError("Ranges cannot be empty")
    
    if len(ranges) > 5:
        raise ValueError("Maximum 5 ranges allowed")
    
    # Validate each range
    for i, r in enumerate(ranges):
        if not isinstance(r, list) or len(r) != 2:
            raise ValueError(f"Range {i} must be a list of [start, end]")
        
        start, end = r
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Range {i} values must be integers")
        
        if start < 0 or end < 0:
            raise ValueError(f"Range {i} values must be non-negative")
        
        if start >= end:
            raise ValueError(f"Range {i}: start ({start}) must be less than end ({end})")
    
    # Check for overlaps
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            start1, end1 = ranges[i]
            start2, end2 = ranges[j]
            
            # Check if ranges overlap
            if not (end1 <= start2 or end2 <= start1):
                raise ValueError(
                    f"Ranges {i} [{start1}, {end1}] and {j} [{start2}, {end2}] overlap"
                )


def ranges_to_task_id_set(ranges: List[List[int]]) -> Set[int]:
    """Convert multi-range to a set of task IDs.

    Args:
        ranges: List of [start, end] pairs (end is exclusive)

    Returns:
        Set of all task IDs in the ranges
    """
    task_ids = set()
    for start, end in ranges:
        task_ids.update(range(start, end))
    return task_ids


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
    
    # Environment Configuration Management
    
    async def get_sampling_environments(self) -> List[str]:
        """Get list of environments for task generation (sampling).
        
        Returns:
            List of environment names where enabled_for_sampling=true
        """
        environments = await self.get_param_value('environments', default={})
        return [
            env_name
            for env_name, env_config in environments.items()
            if env_config.get('enabled_for_sampling', False)
        ]
    
    async def set_sampling_environments(
        self, envs: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Set sampling environments list.
        
        This method updates the enabled_for_sampling flag for environments.
        
        Args:
            envs: List of environment names to enable for sampling
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        environments = await self.get_param_value('environments', default={})
        
        # Update enabled_for_sampling for all environments
        for env_name in environments:
            environments[env_name]['enabled_for_sampling'] = env_name in envs
        
        return await self.set_param(
            param_name='environments',
            param_value=environments,
            param_type='dict',
            description='Environment configurations (sampling, scoring, ranges)',
            updated_by=updated_by
        )
    
    async def get_active_environments(self) -> List[str]:
        """Get list of environments for score calculation (active).
        
        Allows transition period when adding new environments:
        - New env added to sampling list first (enabled_for_sampling=true)
        - After sufficient sampling, added to active list (enabled_for_scoring=true)
        
        Returns:
            List of environment names where enabled_for_scoring=true
        """
        environments = await self.get_param_value('environments', default={})
        return [
            env_name
            for env_name, env_config in environments.items()
            if env_config.get('enabled_for_scoring', False)
        ]
    
    async def set_active_environments(
        self, envs: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Set active environments list for scoring.
        
        This method updates the enabled_for_scoring flag for environments.
        
        Args:
            envs: List of environment names to enable for scoring
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        environments = await self.get_param_value('environments', default={})
        
        # Update enabled_for_scoring for all environments
        for env_name in environments:
            environments[env_name]['enabled_for_scoring'] = env_name in envs
        
        return await self.set_param(
            param_name='environments',
            param_value=environments,
            param_type='dict',
            description='Environment configurations (sampling, scoring, ranges)',
            updated_by=updated_by
        )
    
    async def get_env_task_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get task_id ranges for each environment.
        
        Extracts range information from environments config.
        
        Format (multi-range 2D array):
        {
            "affine:sat": {
                "sampling_range": [[0, 1000]],  # Single range for task generation
                "scoring_range": [[0, 1000]]     # Single range for scoring
            },
            "affine:abd": {
                "sampling_range": [[0, 100], [200, 300]],  # Multiple discontinuous ranges
                "scoring_range": [[0, 300]]                 # Smaller range during transition
            }
        }
        
        Returns:
            Dictionary mapping env names to range configurations
        """
        environments = await self.get_param_value('environments', default={})
        
        # Extract range information from environments
        ranges = {}
        for env_name, env_config in environments.items():
            ranges[env_name] = {
                'sampling_range': env_config.get('sampling_range', [[0, 0]]),
                'scoring_range': env_config.get('scoring_range', [[0, 0]])
            }
        
        return ranges
    
    async def set_env_task_ranges(
        self, ranges: Dict[str, Dict[str, Any]], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Set task_id ranges for environments.
        
        This method updates the sampling_range and scoring_range for environments.
        All ranges must be in 2D array format: [[start1, end1], [start2, end2], ...]
        
        Args:
            ranges: Dictionary mapping env names to range configs with
                   'sampling_range' and 'scoring_range' keys (both must be 2D arrays)
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
            
        Raises:
            ValueError: If range validation fails
        """
        environments = await self.get_param_value('environments', default={})
        
        # Update ranges for specified environments with validation
        for env_name, range_config in ranges.items():
            if env_name in environments:
                if 'sampling_range' in range_config:
                    validate_ranges(range_config['sampling_range'])
                    environments[env_name]['sampling_range'] = range_config['sampling_range']
                if 'scoring_range' in range_config:
                    validate_ranges(range_config['scoring_range'])
                    environments[env_name]['scoring_range'] = range_config['scoring_range']
        
        return await self.set_param(
            param_name='environments',
            param_value=environments,
            param_type='dict',
            description='Environment configurations (sampling, scoring, ranges)',
            updated_by=updated_by
        )
    
    async def get_env_sampling_ranges(self, env: str) -> List[List[int]]:
        """Get sampling ranges for a specific environment (multi-range format).
        
        Args:
            env: Environment name
            
        Returns:
            List of [start, end] pairs for sampling (2D array)
        """
        ranges = await self.get_env_task_ranges()
        env_config = ranges.get(env, {})
        sampling_range = env_config.get('sampling_range', [[0, 0]])
        return sampling_range if isinstance(sampling_range, list) and len(sampling_range) > 0 else [[0, 0]]
    
    async def get_env_scoring_ranges(self, env: str) -> List[List[int]]:
        """Get scoring ranges for a specific environment (multi-range format).
        
        Args:
            env: Environment name
            
        Returns:
            List of [start, end] pairs for scoring (2D array)
        """
        ranges = await self.get_env_task_ranges()
        env_config = ranges.get(env, {})
        scoring_range = env_config.get('scoring_range', [[0, 0]])
        return scoring_range if isinstance(scoring_range, list) and len(scoring_range) > 0 else [[0, 0]]
    
    # Blacklist Configuration Management
    
    async def get_blacklist(self) -> List[str]:
        """Get blacklisted hotkeys from database.
        
        Returns:
            List of blacklisted hotkey strings
        """
        blacklist = await self.get_param_value('miner_blacklist', default=[])
        return blacklist if isinstance(blacklist, list) else []
    
    async def set_blacklist(
        self, hotkeys: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Set blacklisted hotkeys.
        
        Args:
            hotkeys: List of hotkey strings to blacklist
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        # Remove duplicates and empty strings
        unique_hotkeys = list(set(hk.strip() for hk in hotkeys if hk.strip()))
        
        return await self.set_param(
            param_name='miner_blacklist',
            param_value=unique_hotkeys,
            param_type='list',
            description='Blacklisted miner hotkeys',
            updated_by=updated_by
        )
    
    async def add_to_blacklist(
        self, hotkeys: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Add hotkeys to blacklist.
        
        Args:
            hotkeys: List of hotkey strings to add
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        current_blacklist = await self.get_blacklist()
        updated_blacklist = list(set(current_blacklist + hotkeys))
        return await self.set_blacklist(updated_blacklist, updated_by)
    
    async def remove_from_blacklist(
        self, hotkeys: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Remove hotkeys from blacklist.
        
        Args:
            hotkeys: List of hotkey strings to remove
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        current_blacklist = await self.get_blacklist()
        updated_blacklist = [hk for hk in current_blacklist if hk not in hotkeys]
        return await self.set_blacklist(updated_blacklist, updated_by)