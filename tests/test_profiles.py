
import pytest
import json
import os
from pathlib import Path
from affine.config import ConfigManager, ConfigError
from affine.utils.api_client import CLIAPIClient

@pytest.fixture
def clean_config(tmp_path, monkeypatch):
    """Fixture to ensure clean config environment."""
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    
    # Mock Path.home() and Path.cwd()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(Path, "cwd", lambda: cwd)
    
    # Reset singleton
    ConfigManager._instance = None
    ConfigManager._config = None
    ConfigManager._active_profile_name = None
    ConfigManager._cli_profile_override = None
    
    return ConfigManager()

def test_config_precedence(clean_config, tmp_path):
    # Global config
    home_config = tmp_path / "home" / ".affine" / "config.json"
    home_config.parent.mkdir(parents=True, exist_ok=True)
    with open(home_config, "w") as f:
        json.dump({
            "profiles": {
                "default": {"base_url": "https://global.com"}
            }
        }, f)
        
    config = ConfigManager()
    assert config.get_profile("default").base_url == "https://global.com"
    
    # Project config should override
    cwd_config = tmp_path / "cwd" / "affine.json"
    with open(cwd_config, "w") as f:
        json.dump({
            "profiles": {
                "default": {"base_url": "https://project.com"}
            }
        }, f)
        
    config.reload()
    assert config.get_profile("default").base_url == "https://project.com"

def test_profile_switching(clean_config, tmp_path):
    config = clean_config
    # Create config with two profiles
    home_config = tmp_path / "home" / ".affine" / "config.json"
    home_config.parent.mkdir(parents=True, exist_ok=True)
    with open(home_config, "w") as f:
        json.dump({
            "active_profile": "dev",
            "profiles": {
                "dev": {"base_url": "https://dev.com"},
                "prod": {"base_url": "https://prod.com"}
            }
        }, f)
    
    config.reload()
    assert config.get_profile_name() == "dev"
    assert config.get_profile().base_url == "https://dev.com"
    
    # Switch via setter
    config.set_active_profile("prod")
    assert config.get_profile_name() == "prod"
    assert config.get_profile().base_url == "https://prod.com"
    
    # Verify file updated
    with open(home_config, "r") as f:
        data = json.load(f)
    assert data["active_profile"] == "prod"

def test_default_fallback(clean_config, tmp_path):
    config = clean_config
    # No active profile set, should default to "default"
    home_config = tmp_path / "home" / ".affine" / "config.json"
    home_config.parent.mkdir(parents=True, exist_ok=True)
    with open(home_config, "w") as f:
        json.dump({
            "profiles": {
                "default": {"base_url": "https://default.com"}
            }
        }, f)
    
    config.reload()
    assert config.get_profile_name() == "default"
    assert config.get_profile().base_url == "https://default.com"

def test_invalid_profile_handling(clean_config, tmp_path):
    config = clean_config
    home_config = tmp_path / "home" / ".affine" / "config.json"
    home_config.parent.mkdir(parents=True, exist_ok=True)
    with open(home_config, "w") as f:
        json.dump({
            "active_profile": "non_existent",
            "profiles": {
                "default": {"base_url": "https://default.com"}
            }
        }, f)
    
    config.reload()
    with pytest.raises(ConfigError):
        config.get_profile()

def test_cli_override(clean_config, tmp_path):
    config = clean_config
    home_config = tmp_path / "home" / ".affine" / "config.json"
    home_config.parent.mkdir(parents=True, exist_ok=True)
    with open(home_config, "w") as f:
        json.dump({
            "active_profile": "default",
            "profiles": {
                "default": {"base_url": "https://default.com"},
                "custom": {"base_url": "https://custom.com"}
            }
        }, f)
        
    config.reload()
    config.set_cli_profile("custom")
    assert config.get_profile_name() == "custom"
    assert config.get_profile().base_url == "https://custom.com"

def test_set_profile_value(clean_config, tmp_path):
    config = clean_config
    # Start with empty config
    
    config.set_profile_value("new_profile", "base_url", "https://new.com")
    config.set_profile_value("new_profile", "retry.max_attempts", "5")
    
    profile = config.get_profile("new_profile")
    assert profile.base_url == "https://new.com"
    assert profile.retry.max_attempts == 5

@pytest.mark.asyncio
async def test_api_client_integration(clean_config, tmp_path):
    home_config = tmp_path / "home" / ".affine" / "config.json"
    home_config.parent.mkdir(parents=True, exist_ok=True)
    with open(home_config, "w") as f:
        json.dump({
            "profiles": {
                "test": {
                    "base_url": "https://test.api",
                    "timeout": 42,
                    "retry": {"max_attempts": 2}
                }
            }
        }, f)
    
    clean_config.reload()
    clean_config.set_active_profile("test")
    
    client = CLIAPIClient()
    # It should pick up config
    assert client.base_url == "https://test.api"
    assert client.timeout == 42
    assert client.retry_config.max_attempts == 2
