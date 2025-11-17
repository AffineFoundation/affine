#!/usr/bin/env python3
from __future__ import annotations
__version__ = "0.0.0"

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
from affine.core.config import singleton, get_conf

# --------------------------------------------------------------------------- #
#                                Logging                                      #
# --------------------------------------------------------------------------- #
from affine.core.setup import (
    logger, setup_logging, info, debug, trace
)

# --------------------------------------------------------------------------- #
#                   Data Models (imported from models module)                 #
# --------------------------------------------------------------------------- #
from affine.core.models import (
    Challenge, Response, Miner, Result
)
# --------------------------------------------------------------------------- #
#                   HTTP client (imported from http_client module)            #
# --------------------------------------------------------------------------- #
from affine.core.http_client import _get_client

# --------------------------------------------------------------------------- #
#                   Miners (imported from miners module)                      #
# --------------------------------------------------------------------------- #
from affine.core.miners import miners


# --------------------------------------------------------------------------- #
#                              SDK Exports                                    #
# --------------------------------------------------------------------------- #
# Import SDK functions for easy access (migrated to core/environments.py)
from affine.core.environments import (
    # Factory functions matching the expected API
    SAT_factory as SAT,
    ABD_factory as ABD,
    DED_factory as DED,
    ALFWORLD_factory as ALFWORLD,
    WEBSHOP_factory as WEBSHOP,
    BABYAI_factory as BABYAI,
    SCIWORLD_factory as SCIWORLD,
    TEXTCRAFT_factory as TEXTCRAFT,
)
