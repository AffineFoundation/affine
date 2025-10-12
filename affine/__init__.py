#!/usr/bin/env python3
from __future__ import annotations
__version__ = "0.0.0"
from .utils import *
from .quixand.core.sandbox_manager import get_sandbox
from .sampling import MinerSampler, SamplingOrchestrator, SamplingConfig

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
from .config import NETUID, singleton, get_conf, _get_env_list_from_envvar, ENVS

# --------------------------------------------------------------------------- #
#                       Prometheus & Logging                                  #
# --------------------------------------------------------------------------- #
from .setup import (
    TRACE, METRICS_PORT, METRICS_ADDR, REGISTRY,
    QCOUNT, SCORE, RANK, WEIGHT, LASTSET, NRESULTS, MAXENV, CACHE,
    logger, setup_logging, info, debug, trace
)

# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
from affine.utils.subtensor import get_subtensor

# --------------------------------------------------------------------------- #
#                   Data Models (imported from models module)                 #
# --------------------------------------------------------------------------- #
from .models import (
    BaseEnv, ContainerEnv, AgentGymContainerEnv, AffineContainerEnv,
    Challenge, Evaluation, Response, Miner, Result,
    _SBX_POOL, _SBX_LOCKS, _SBX_SEMS
)

# --------------------------------------------------------------------------- #
#                   S3/Storage helpers (imported from storage module)         #
# --------------------------------------------------------------------------- #
from .storage import dataset, sink_enqueue, sign_results, prune, CACHE_DIR

# --------------------------------------------------------------------------- #
#                   Query client (imported from query module)                 #
# --------------------------------------------------------------------------- #
from .query import query, run, _get_client, LOG_TEMPLATE

# --------------------------------------------------------------------------- #
#                   Miners (imported from miners module)                      #
# --------------------------------------------------------------------------- #
from .miners import get_chute, get_chute_code, get_latest_chute_id, get_weights_shas, miners

# --------------------------------------------------------------------------- #
#                   Validator (imported from validator module)                #
# --------------------------------------------------------------------------- #
from .validator import get_weights, retry_set_weights, validate_api_key, check_env_variables, _set_weights_with_confirmation

# --------------------------------------------------------------------------- #
#                   CLI (imported from cli module)                            #
# --------------------------------------------------------------------------- #
from .cli import cli

# --------------------------------------------------------------------------- #
#                              SDK Exports                                    #
# --------------------------------------------------------------------------- #
# Import SDK functions for easy access
from .tasks import (
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
