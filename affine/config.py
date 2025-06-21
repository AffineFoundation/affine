"""
Configuration management for the Affine framework.
Provides consistent, elegant configuration loading from multiple sources.
"""

import os
import logging
import configparser
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Enhanced logging with TRACE level
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

def _trace(self, msg, *args, **kwargs):
    """Add TRACE level to logger."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)

logging.Logger.trace = _trace


def setup_logging(verbosity: int = 0):
    """
    Setup logging with different verbosity levels.
    
    Args:
        verbosity: 0=CRITICAL+, 1=INFO, 2=DEBUG, 3+=TRACE
    """
    if verbosity >= 3: 
        level = TRACE
    elif verbosity == 2: 
        level = logging.DEBUG
    elif verbosity == 1: 
        level = logging.INFO
    else: 
        level = logging.CRITICAL + 1
    
    # Silence noisy loggers
    for noisy_logger in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class ConfigManager:
    """Centralized configuration management with elegant fallback logic."""
    
    def __init__(self):
        self._load_env_files()
        self._config = self._load_ini_files()
        
    def _load_env_files(self) -> None:
        """Load environment files in order of priority."""
        env_files = [
            Path.home() / ".affine" / "config.env",
            Path.cwd() / ".env"
        ]
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file, override=False)
    
    def _load_ini_files(self) -> configparser.ConfigParser:
        """Load INI configuration files."""
        config = configparser.ConfigParser()
        ini_files = [
            Path.home() / ".chutes" / "config.ini",
            Path.home() / ".affine" / "config.ini"
        ]
        existing_files = [f for f in ini_files if f.exists()]
        if existing_files:
            config.read(existing_files)
        return config
    
    def get(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """Get configuration value with elegant fallback logic."""
        # 1. Environment variables (highest priority)
        value = os.getenv(key)
        if value:
            return value
            
        # 2. INI files with key mapping
        ini_value = self._get_from_ini(key)
        if ini_value:
            return ini_value
            
        # 3. Default value
        if default is not None:
            return default
            
        # 4. Handle required values
        if required:
            raise ValueError(f"Required configuration '{key}' not found")
            
        return None
    
    def _get_from_ini(self, key: str) -> Optional[str]:
        """Get value from INI files with standardized key mapping."""
        # Standardized key mappings for consistency
        key_mappings = {
            "CHUTES_API_KEY": ["chutes_api_key", "api_key"],
            "HF_TOKEN": ["hf_token", "hugging_face_token", "huggingface_token"],
            "HF_USER": ["hf_user", "huggingface_user"],
            "CHUTES_USER": ["chutes_user", "chute_user", "username"],
            "BT_COLDKEY": ["bt_coldkey", "wallet_cold", "coldkey"],
            "BT_HOTKEY": ["bt_hotkey", "wallet_hot", "hotkey"],
            "CHUTES_IMAGE": ["chutes_image", "image"],
        }
        
        possible_keys = key_mappings.get(key, [key.lower()])
        
        for section_name in self._config.sections():
            section = self._config[section_name]
            for possible_key in possible_keys:
                if possible_key in section:
                    return section[possible_key]
        
        return None
    
    def get_required(self, key: str) -> str:
        """Get required configuration value or raise error."""
        return self.get(key, required=True)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        result = {}
        
        # Add environment variables
        for key in os.environ:
            result[key] = os.environ[key]
            
        # Add INI values
        for section_name in self._config.sections():
            for key, value in self._config[section_name].items():
                result[f"{section_name}.{key}"] = value
                
        return result


# Global config instance
config = ConfigManager()

# Convenience functions for backward compatibility
def get_conf(key: str) -> str:
    """Get required configuration value (legacy function)."""
    return config.get_required(key)

def get_config_value(key: str, cli_value: Optional[str] = None) -> Optional[str]:
    """Get configuration value with CLI override (legacy function)."""
    return cli_value or config.get(key) 