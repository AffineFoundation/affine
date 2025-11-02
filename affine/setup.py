import os
import logging
from dotenv import load_dotenv
from typing import Tuple, Type

load_dotenv(override=True)

NETUID = 120

TRACE = 5

# Import environment classes for type-safe configuration
# Note: This import must be after other imports to avoid circular dependencies
def get_enabled_envs():
    """Get enabled environment classes. Imported here to avoid circular dependency."""
    from affine.tasks import (
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
    logging.getLogger("affinetes").setLevel(logging.WARNING)

    # Set affine logger level
    logging.getLogger("affine").setLevel(level)

def info():
    setup_logging(1)

def debug():
    setup_logging(2)

def trace():
    setup_logging(3)