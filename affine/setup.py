import os
import logging
from dotenv import load_dotenv
from typing import Tuple

load_dotenv(override=True)

NETUID = 120

TRACE = 5

ENVS: Tuple[str, ...] = (
    "agentgym:webshop",
    "agentgym:alfworld",
    "agentgym:babyai",
    "agentgym:sciworld",
    "agentgym:textcraft",
    "affine:sat",
    "affine:ded",
    "affine:abd",
)

logging.addLevelName(TRACE, "TRACE")

def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)

logging.Logger.trace = _trace
logger = logging.getLogger("affine")

def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("affine").setLevel(level)

def info():
    setup_logging(1)

def debug():
    setup_logging(2)

def trace():
    setup_logging(3)