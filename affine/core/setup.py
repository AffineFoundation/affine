import os
import logging
from dotenv import load_dotenv
from typing import Tuple, Type, Optional

load_dotenv(override=True)

NETUID = 120

# Wallet initialization
def get_wallet() -> Optional["bt.wallet"]:
    """Get bittensor wallet from environment variables.
    
    Returns:
        Bittensor wallet or None if not configured
    """
    try:
        import bittensor as bt
        
        coldkey = os.getenv("BT_WALLET_COLD", "default")
        hotkey = os.getenv("BT_WALLET_HOT", "default")
        
        return bt.wallet(name=coldkey, hotkey=hotkey)
    
    except Exception as e:
        logging.getLogger("affine").warning(f"Failed to initialize wallet: {e}")
        return None

# Global wallet instance (lazy initialization)
_wallet_instance: Optional["bt.wallet"] = None

def get_wallet_instance() -> Optional["bt.wallet"]:
    """Get or create global wallet instance."""
    global _wallet_instance
    
    if _wallet_instance is None:
        _wallet_instance = get_wallet()
    
    return _wallet_instance

# Export wallet for backward compatibility
wallet = get_wallet_instance()

TRACE = 5

# Import environment classes for type-safe configuration
# Note: This import must be after other imports to avoid circular dependencies
def get_enabled_envs():
    """Get enabled environment classes. Imported here to avoid circular dependency."""
    from affine.core.environments import (
        WEBSHOP,
        ALFWORLD,
        BABYAI,
        SCIWORLD,
        TEXTCRAFT,
        SAT,
        DED,
        ABD,
        BaseSDKEnv,
    )
    
    # Type-safe environment configuration using actual classes
    ENABLED_ENVS: Tuple[Type[BaseSDKEnv], ...] = (
        WEBSHOP,
        ALFWORLD,
        BABYAI,
        SCIWORLD,
        TEXTCRAFT,
        SAT,
        DED,
        ABD,
    )
    
    return ENABLED_ENVS

def get_env_names() -> Tuple[str, ...]:
    """Get enabled environment names as strings."""
    return tuple(env_class._env_name for env_class in get_enabled_envs())


for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore", "httpx", "httpcore", "docker", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# Configure affinetes logger to prevent duplicate logs
# affinetes has its own handler, so disable propagation to root logger
affinetes_logger = logging.getLogger("affinetes")
affinetes_logger.setLevel(logging.WARNING)
affinetes_logger.propagate = False

logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

logging.addLevelName(TRACE, "TRACE")


def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)

logging.Logger.trace = _trace
logger = logging.getLogger("affine")

def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1

    # Silence noisy loggers
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore", "httpx", "httpcore", "docker", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set affinetes logger to WARNING to reduce noise
    # Disable propagate to prevent duplicate logs (affinetes has its own handler)
    affinetes_logger = logging.getLogger("affinetes")
    affinetes_logger.setLevel(logging.WARNING)
    affinetes_logger.propagate = False

    # Set affine logger level
    logging.getLogger("affine").setLevel(level)

def info():
    setup_logging(1)

def debug():
    setup_logging(2)

def trace():
    setup_logging(3)