#!/usr/bin/env python3
from __future__ import annotations
__version__ = "0.0.0"

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
    Miner, Result
)

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
    DED_V2_factory as DED,
    ABD_V2_factory as ABD,
    DED_V2_factory as DED_V2,
    ABD_V2_factory as ABD_V2,
    ALFWORLD_factory as ALFWORLD,
    WEBSHOP_factory as WEBSHOP,
    BABYAI_factory as BABYAI,
    SCIWORLD_factory as SCIWORLD,
    TEXTCRAFT_factory as TEXTCRAFT,
    TEXTCRAFT_factory as TEXTCRAFT,
    CDE_factory as CDE,
    LGC_factory as LGC,
    MTH_factory as MTH,
    SCI_factory as SCI,
    list_available_environments,
)

# Create tasks namespace for backward compatibility
class _TasksNamespace:
    """Namespace for task-related functions"""
    list_available_environments = staticmethod(list_available_environments)

tasks = _TasksNamespace()
