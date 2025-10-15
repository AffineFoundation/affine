#!/usr/bin/env python3
from __future__ import annotations
__version__ = "0.0.0"

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
from affine.config import singleton, get_conf

# --------------------------------------------------------------------------- #
#                                Logging                                      #
# --------------------------------------------------------------------------- #
from affine.setup import (
    logger, setup_logging, info, debug, trace
)

# --------------------------------------------------------------------------- #
#                   Data Models (imported from models module)                 #
# --------------------------------------------------------------------------- #
from affine.models import (
    Challenge, Evaluation, Response, Miner, Result
)
# --------------------------------------------------------------------------- #
#                   Query client (imported from query module)                 #
# --------------------------------------------------------------------------- #
from affine.query import query, _get_client

# --------------------------------------------------------------------------- #
#                   Miners (imported from miners module)                      #
# --------------------------------------------------------------------------- #
from affine.miners import miners

# --------------------------------------------------------------------------- #
#                   CLI (imported from cli module)                            #
# --------------------------------------------------------------------------- #
from affine.cli import cli

# --------------------------------------------------------------------------- #
#                               dataset                                       #
# --------------------------------------------------------------------------- #
from affine.storage import (
    FOLDER, BUCKET, ACCESS, SECRET, ENDPOINT, PUBLIC_READ, R2_PUBLIC_BASE
)


# --------------------------------------------------------------------------- #
#                              SDK Exports                                    #
# --------------------------------------------------------------------------- #
# Import SDK functions for easy access
from affine.tasks import (
    # Factory functions matching the expected API
    SAT_factory as SAT,
    ABD_factory as ABD,
    DED_factory as DED,
    HVM_factory as HVM,
    ELR_factory as ELR,
    ALFWORLD_factory as ALFWORLD,
    WEBSHOP_factory as WEBSHOP,
    BABYAI_factory as BABYAI,
    SCIWORLD_factory as SCIWORLD,
    TEXTCRAFT_factory as TEXTCRAFT,
)
