
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from affine.core.setup import logger

class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_seconds: float = 1.0

class ProfileConfig(BaseModel):
    base_url: str
    timeout: int = 30
    retry: RetryConfig = Field(default_factory=RetryConfig)
    mock: bool = False
    environment_overrides: Optional[Dict[str, Any]] = None

class AffineConfig(BaseModel):
    active_profile: Optional[str] = None
    profiles: Dict[str, ProfileConfig] = Field(default_factory=dict)

class ConfigError(Exception):
    """Base class for configuration errors."""
    pass

class ConfigManager:
    _instance = None
    _config: Optional[AffineConfig] = None
    _active_profile_name: Optional[str] = None
    _cli_profile_override: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure we only load once if singleton is reused/re-initialized
        if self._config is None:
            self.reload()

    def reload(self):
        """Reload configuration from files."""
        self._config = self._load_config()

    def set_cli_profile(self, profile_name: str):
        """Set the profile override from CLI arguments."""
        self._cli_profile_override = profile_name

    def _load_config(self) -> AffineConfig:
        """Load and merge configuration from global and project files."""
        
        # 1. Global config
        global_path = Path.home() / ".affine" / "config.json"
        global_data = {}
        if global_path.exists():
            try:
                with open(global_path, "r") as f:
                    global_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in global config ({global_path}): {e}")
            except Exception as e:
                logger.warning(f"Failed to read global config: {e}")

        # 2. Project config
        project_path = Path.cwd() / "affine.json"
        project_data = {}
        if project_path.exists():
            try:
                with open(project_path, "r") as f:
                    project_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in project config ({project_path}): {e}")
            except Exception as e:
                # If we can't read project config, that's possibly fine, but if it exists we should probably warn
                logger.warning(f"Failed to read project config: {e}")

        # Merge strategy: Project overrides Global
        # Deep merge for profiles is not strictly required by spec ("If both exist, project config MUST override global config")
        # usually means keys in project overwrite keys in global.
        # But for 'profiles', do we merge the dictionary or replace it? 
        # Usually detailed merge is better, but simple dict update is easier.
        # Let's do a shallow merge of the top level keys.
        
        merged_data = global_data.copy()
        
        # Merge 'profiles' specifically to allow project to add/override individual profiles
        if "profiles" in project_data:
            if "profiles" not in merged_data:
                merged_data["profiles"] = {}
            # We don't want to wipe out global profiles if project defines just one
            # So we update the profiles dict
            if isinstance(project_data["profiles"], dict) and isinstance(merged_data["profiles"], dict):
                merged_data["profiles"].update(project_data["profiles"])
            else:
                # If types mismatch (shouldn't happen with valid schema), let project win
                merged_data["profiles"] = project_data["profiles"]
        
        # Update other top-level keys (like active_profile)
        for k, v in project_data.items():
            if k != "profiles":
                merged_data[k] = v

        try:
            return AffineConfig(**merged_data)
        except ValidationError as e:
            # format validation error
            msg = "Configuration validation failed:\n"
            for err in e.errors():
                loc = ".".join(str(l) for l in err['loc'])
                msg += f" - {loc}: {err['msg']}\n"
            raise ConfigError(msg)

    def get_profile_name(self) -> str:
        """Resolve the active profile name."""
        if self._cli_profile_override:
            return self._cli_profile_override
        
        if self._config.active_profile:
            return self._config.active_profile
            
        return "default"

    def get_profile(self, profile_name: Optional[str] = None) -> ProfileConfig:
        """Get the resolved profile configuration."""
        if self._config is None:
            self.reload()
            
        name = profile_name or self.get_profile_name()
        
        if name not in self._config.profiles:
             # Fallback/Default behavior if strictly no config found is requested
             # Spec says: "If resolved profile does not exist, raise a clear error."
             
             # However, to be nice to new users, if the config is completely empty (no profiles),
             # maybe we should check if they passed environment variables for legacy support?
             # Spec says: "ALL SDK entrypoints must resolve configuration internally."
             # "User code must NOT pass... manually."
             # This implies we should enforce the config system.
             # But if I break existing env var setups, that might be bad?
             # User requirements didn't say "support env vars".
             # Actually, item 4 says: "Configuration is read once and cached per process."
             
             # I will stick to the spec. RAISE ERROR.
             raise ConfigError(f"Profile '{name}' not found in configuration. Available profiles: {list(self._config.profiles.keys())}")

        return self._config.profiles[name]

    def get_all_profiles(self) -> Dict[str, ProfileConfig]:
        if self._config is None:
            self.reload()
        return self._config.profiles

    def set_active_profile(self, profile_name: str):
        """Set the active profile in the appropriate config file."""
        # We prefer writing to project config if it exists, otherwise global?
        # Or always global? "af config use <profile>" calls this.
        # Usually 'use' implies local context if local config exists, or global if not?
        # Let's check where the config came from.
        # Simple heuristic: If ./affine.json exists, modify it. Else modify ~/.affine/config.json
        
        target_file = Path.cwd() / "affine.json"
        if not target_file.exists():
            target_file = Path.home() / ".affine" / "config.json"
            
        # Ensure directory exists for global
        if target_file == Path.home() / ".affine" / "config.json":
            target_file.parent.mkdir(parents=True, exist_ok=True)

        # Read existing file to preserve other fields not managed by us?
        # Or just read/update/write.
        data = {}
        if target_file.exists():
            try:
                with open(target_file, "r") as f:
                    data = json.load(f)
            except Exception:
                pass # start fresh if corrupt
        
        data["active_profile"] = profile_name
        
        with open(target_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Reload to reflect changes
        self.reload()

    def set_profile_value(self, profile_name: str, key: str, value: str):
        """Set a value in a profile (dot notation supported for nested keys)."""
        # Similar logic: prioritize local config if it exists or if we are creating it?
        # If the profile exists in local, update local. If in global, update global.
        # If in neither, default to... local?
        
        project_path = Path.cwd() / "affine.json"
        global_path = Path.home() / ".affine" / "config.json"
        
        # Load both raw
        project_data = {}
        if project_path.exists():
            with open(project_path, "r") as f:
                project_data = json.load(f)
                
        global_data = {}
        if global_path.exists():
            with open(global_path, "r") as f:
                global_data = json.load(f)
        
        # Determine target
        target_path = global_path
        target_data = global_data
        
        # If profile is explicitly in project data, use project
        if "profiles" in project_data and profile_name in project_data["profiles"]:
            target_path = project_path
            target_data = project_data
        elif "profiles" in global_data and profile_name in global_data["profiles"]:
             target_path = global_path
             target_data = global_data
        else:
             # Create in global by default unless project file already exists?
             # Let's say default to global to keep project overrides clean?
             # Or default to project if we invoke this in a project dir?
             # Let's default to global for "user preferences" style, but "project config" might be shared in repo.
             # Let's go with global default for now, unless project file exists.
             if project_path.exists():
                 target_path = project_path
                 target_data = project_data
             else:
                 # Ensure global dir
                 global_path.parent.mkdir(parents=True, exist_ok=True)
                 target_path = global_path
                 target_data = global_data

        if "profiles" not in target_data:
            target_data["profiles"] = {}
        if profile_name not in target_data["profiles"]:
            target_data["profiles"][profile_name] = {"base_url": ""} # Initialize with required field placeholder if needed, though validated on load

        # Handle nested keys (e.g. retry.max_attempts)
        keys = key.split('.')
        current = target_data["profiles"][profile_name]
        
        # We need to be careful about types. Value is string.
        # We should infer type based on schema?
        # Pydantic doesn't help with raw dict mutation easily without schema.
        # Let's try to infer int/bool/float.
        
        def infer_type(v):
            if v.lower() == 'true': return True
            if v.lower() == 'false': return False
            try:
                return int(v)
            except ValueError:
                try:
                    return float(v)
                except ValueError:
                    return v

        typed_value = infer_type(value)

        for i, k in enumerate(keys[:-1]):
            if k not in current:
                current[k] = {}
            current = current[k]
            if not isinstance(current, dict):
                 # Schema violation likely if we try to treat a leaf as node
                 raise ConfigError(f"Key path conflict at '{k}'")

        current[keys[-1]] = typed_value
        
        # Write back
        with open(target_path, "w") as f:
            json.dump(target_data, f, indent=2)
            
        self.reload()

# Global singleton
config = ConfigManager()
