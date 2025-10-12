#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import os
import re
import sys
import math
import json
import time
import click
import socket
import random
import hashlib
import aiohttp
import asyncio
import logging
import requests
import textwrap
import traceback
import itertools
import atexit
from .utils import *
from math import comb
import datetime as dt
from tqdm import tqdm
import bittensor as bt
import datasets as hf_ds                    
from pathlib import Path
from tqdm.asyncio import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
from typing import AsyncIterator
from urllib.parse import urlparse
from huggingface_hub import HfApi
from botocore.config import Config
from collections import defaultdict
from abc import ABC, abstractmethod
from pydantic import root_validator
from aiohttp import ClientConnectorError
from aiobotocore.session import get_session
from huggingface_hub import snapshot_download
from bittensor.core.errors import MetadataError
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable
__version__ = "0.0.0"
from .quixand.core.sandbox_manager import get_sandbox
from .sampling import MinerSampler, SamplingOrchestrator, SamplingConfig

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
NETUID = 120
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")
_SINGLETON_CACHE = {}
def singleton(key:str, factory):
    """Create a singleton factory function that creates an object only once."""
    def get_instance():
        if key not in _SINGLETON_CACHE:
            _SINGLETON_CACHE[key] = factory()
        return _SINGLETON_CACHE[key]
    return get_instance

# --------------------------------------------------------------------------- #
#                       Prometheus                         #
# --------------------------------------------------------------------------- #
from prometheus_client import Counter, CollectorRegistry, start_http_server, Gauge
METRICS_PORT   = int(os.getenv("AFFINE_METRICS_PORT", "8000"))
METRICS_ADDR   = os.getenv("AFFINE_METRICS_ADDR", "0.0.0.0")
REGISTRY       = CollectorRegistry(auto_describe=True)
QCOUNT  = Counter("qcount", "qcount", ["model"], registry=REGISTRY)
SCORE   = Gauge( "score", "score", ["uid", "env"], registry=REGISTRY)
RANK    = Gauge( "rank", "rank", ["uid", "env"], registry=REGISTRY)
WEIGHT  = Gauge( "weight", "weight", ["uid"], registry=REGISTRY)
LASTSET = Gauge( "lastset", "lastset", registry=REGISTRY)
NRESULTS = Gauge( "nresults", "nresults", registry=REGISTRY)
MAXENV = Gauge("maxenv", "maxenv", ["env"], registry=REGISTRY)
CACHE = Gauge( "cache", "cache", registry=REGISTRY)


# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    if not getattr(setup_logging, "_prom_started", False):
        try: start_http_server(METRICS_PORT, METRICS_ADDR, registry=REGISTRY)
        except: pass
        setup_logging._prom_started = True
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
def info():setup_logging(1)
def debug():setup_logging(2)
def trace():setup_logging(3)

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key); 
    if not v and default is None:
        raise ValueError(f"{key} not set.\nYou must set env var: {key} in .env")
    return v or default


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

# Central env registry (Quixand‑only)
def _get_env_list_from_envvar() -> Tuple[str, ...]:
    spec = os.getenv("AFFINE_ENV_LIST", "").strip()
    if not spec:
        return tuple()
    env_names: list[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        env_names.append(tok)
    return tuple(env_names)

# Keep variable name ENVS for scoring logic; values are env name strings
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
