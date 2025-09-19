
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
from .utils import *
from math import comb
from dataclasses import dataclass
import datetime as dt
from tqdm import tqdm
import bittensor as bt
import datasets as hf_ds                    
from pathlib import Path
from tqdm.asyncio import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
from types import SimpleNamespace
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
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable, Iterable, DefaultDict, MutableMapping, Collection
__version__ = "0.0.0"

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

# Model gating check cache
MODEL_GATING_CACHE = {}  # {model_id: (is_gated, last_checked)}
# Replace global loop-bound lock with per-event-loop lazy locks to avoid cross-loop errors
_GATING_LOCKS: Dict[int, asyncio.Lock] = {}
GATING_TTL = 3600  # 60 min

def _get_gating_lock() -> asyncio.Lock:
    """Return an asyncio.Lock bound to the current running loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Fallback if called when no loop is running yet
        loop = asyncio.get_event_loop()
    key = id(loop)
    lock = _GATING_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _GATING_LOCKS[key] = lock
    return lock

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

async def check_model_gated(model_id: str, revision: Optional[str] = None) -> Optional[bool]:
    async with _get_gating_lock():
        now = time.time()
        cached = MODEL_GATING_CACHE.get(model_id)
        if cached and now - cached[1] < GATING_TTL:
            return cached[0]
        try:
            r = await asyncio.to_thread(requests.get, f"https://huggingface.co/api/models/{model_id}", timeout=5)
            if r.status_code == 200:
                is_gated = r.json().get("gated", False)
                if revision:
                    try:
                        ok = await asyncio.to_thread(lambda: bool(HfApi(token=os.getenv("HF_TOKEN")).repo_info(repo_id=model_id, revision=revision, repo_type="model")))
                        if not ok: is_gated = True
                    except:
                        pass
                MODEL_GATING_CACHE[model_id] = (is_gated, now)
                return is_gated
        except Exception as e:
            logger.trace(f"Gate check failed for {model_id}: {e}")
        if cached:
            MODEL_GATING_CACHE[model_id] = (cached[0], now)
            return cached[0]
        return None


# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.trace("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor( get_conf('SUBTENSOR_ENDPOINT', default='finney') )
        try:
            await SUBTENSOR.initialize()
            logger.trace("Connected")
        except Exception as e:
            logger.warning(f"Failed to initialize subtensor: {e}, falling back to {'wss://lite.sub.latent.to:443'}")
            SUBTENSOR = bt.async_subtensor( get_conf('SUBTENSOR_FALLBACK', default="wss://lite.sub.latent.to:443") )
            await SUBTENSOR.initialize()
            logger.trace("Connected to fallback")
    return SUBTENSOR

# --------------------------------------------------------------------------- #
#                           Base‑level data models                            #
# --------------------------------------------------------------------------- #
def _truncate(t: Optional[str], max_len: int = 80) -> str:
    return "" if not t else textwrap.shorten(t, width=max_len, placeholder="…")

class BaseEnv(BaseModel, ABC):
    """Abstract competition environment."""
    class Config: arbitrary_types_allowed = True
    @property
    def name(self) -> str: return self.__class__.__name__
    def __hash__(self):     return hash(self.name)
    def __repr__(self):     return self.name
    # API expected from concrete envs
    @abstractmethod
    async def generate(self) -> "Challenge": ...
    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: "Response") -> "Evaluation": ...

# --------------------------------------------------------------------------- #
#                         Models with new (de)serialisation                   #
# --------------------------------------------------------------------------- #
class Challenge(BaseModel):
    env:  BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    challenge_id: Optional[str] = None
    @root_validator(pre=True)
    def set_challenge_id(cls, values):
        if "challenge_id" not in values or values["challenge_id"] is None:
            env = values["env"]
            prompt = values["prompt"]
            extra = values.get("extra", {})
            if not isinstance(env, str): env = env.name
            base_dict = { "env": env,"prompt": prompt, "extra": extra}
            canonical = json.dumps(base_dict, sort_keys=True, separators=(",", ":"))
            cid = hashlib.sha256(canonical.encode()).hexdigest()
            values["challenge_id"] = cid
        return values
    @validator("env", pre=True)
    def _parse_env(cls, v):
        from .envs import ENVS as _ENVS
        return _ENVS[v]() if isinstance(v, str) else v
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    async def evaluate(self, resp: "Response") -> "Evaluation":
        return await self.env.evaluate(self, resp)
    def __repr__(self):
        return f"<Challenge env={self.env.name!r} prompt={_truncate(self.prompt)!r}>"
    __str__ = __repr__


class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)
    @validator("env", pre=True)
    def _parse_env(cls, v):
        from .envs import ENVS as _ENVS
        return _ENVS[v]() if isinstance(v, str) else v
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    def __repr__(self):
        ex = {k: _truncate(str(v)) for k, v in self.extra.items()}
        return f"<Evaluation env={self.env.name!r} score={self.score:.4f} extra={ex!r}>"
    __str__ = __repr__

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]
    success: bool
    def __repr__(self):
        return (f"<Response model={self.model!r} success={self.success} "
                f"latency={self.latency_seconds:.3f}s attempts={self.attempts} "
                f"response={_truncate(self.response)!r} error={_truncate(self.error)!r}>")
    __str__ = __repr__

class Miner(BaseModel):
    uid: int; hotkey: str; model: Optional[str] = None
    revision: Optional[str] = None; block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None
    slug: Optional[str] = None
    

class Result(BaseModel):
    version: str = __version__
    signature: str = ""
    hotkey: str = ""
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation
    def sign(self, wallet):
        self.hotkey = wallet.hotkey.ss58_address
        self.signature = (wallet.hotkey.sign( data = str(self.challenge) )).hex()
    def verify( self ) -> bool:
        return bt.Keypair(ss58_address=self.hotkey).verify( data = str(self.challenge), signature = bytes.fromhex( self.signature) )
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    def __repr__(self): return f"<Result {self.miner.uid=} {self.challenge.env.name=} score={self.evaluation.score:.4f}>"
    __str__ = __repr__

# --------------------------------------------------------------------------- #
#                       Online IRT (2PL) utilities                            #
# --------------------------------------------------------------------------- #


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class _EnvParams:
    discrimination: float = 1.0
    difficulty: float = 0.0
    count: int = 0


@dataclass
class _ChallengeParams:
    bias: float = 0.0
    count: int = 0


@dataclass
class _AbilityParams:
    theta: float = 0.0
    count: int = 0


class OnlineIRT2PL:
    """Online multi-facet two-parameter logistic estimator."""

    def __init__(
        self,
        env_names: Iterable[str] = (),
        *,
        theta_prior: float = 0.0,
        discrimination_prior: float = 1.0,
        difficulty_prior: float = 0.0,
        challenge_prior: float = 0.0,
        theta_lr: float = 0.2,
        difficulty_lr: float = 0.05,
        discrimination_lr: float = 0.01,
        challenge_lr: float = 0.03,
        theta_clip: float = 6.0,
        diff_clip: float = 6.0,
        disc_bounds: Tuple[float, float] = (0.25, 5.0),
        challenge_decay: float = 0.01,
    ) -> None:
        self._theta_prior = theta_prior
        self._discrimination_prior = discrimination_prior
        self._difficulty_prior = difficulty_prior
        self._challenge_prior = challenge_prior
        self._theta_lr = theta_lr
        self._difficulty_lr = difficulty_lr
        self._discrimination_lr = discrimination_lr
        self._challenge_lr = challenge_lr
        self._theta_clip = theta_clip
        self._diff_clip = diff_clip
        self._disc_bounds = disc_bounds
        self._challenge_decay = challenge_decay

        self._env_state: Dict[str, _EnvParams] = {
            name: _EnvParams(discrimination=discrimination_prior, difficulty=difficulty_prior)
            for name in env_names
        }
        self._challenge_state: Dict[str, _ChallengeParams] = {}
        self._ability_state: Dict[Tuple[str, str], _AbilityParams] = {}

    def _ensure_env(self, env: str) -> _EnvParams:
        env_state = self._env_state.get(env)
        if env_state is None:
            env_state = _EnvParams(
                discrimination=self._discrimination_prior,
                difficulty=self._difficulty_prior,
            )
            self._env_state[env] = env_state
        return env_state

    def _ensure_challenge(self, challenge_id: Optional[str]) -> Optional[_ChallengeParams]:
        if not challenge_id:
            return None
        chal = self._challenge_state.get(challenge_id)
        if chal is None:
            chal = _ChallengeParams(bias=self._challenge_prior)
            self._challenge_state[challenge_id] = chal
        return chal

    def _ensure_ability(self, miner_key: Tuple[str, str]) -> _AbilityParams:
        ability = self._ability_state.get(miner_key)
        if ability is None:
            ability = _AbilityParams(theta=self._theta_prior)
            self._ability_state[miner_key] = ability
        return ability

    def observe(
        self,
        *,
        miner_key: Tuple[str, str],
        env: str,
        success: float,
        challenge_id: Optional[str] = None,
        weight: float = 1.0,
    ) -> float:
        ability = self._ensure_ability(miner_key)
        env_state = self._ensure_env(env)
        challenge_state = self._ensure_challenge(challenge_id)

        bias = env_state.difficulty
        if challenge_state is not None:
            challenge_state.bias *= (1.0 - self._challenge_decay)
            bias += challenge_state.bias

        eta = env_state.discrimination * (ability.theta - bias)
        prob = _sigmoid(eta)
        residual = (success - prob) * weight

        ability.count += 1
        ability_step = self._theta_lr / math.sqrt(max(1, ability.count))
        ability.theta += ability_step * env_state.discrimination * residual
        ability.theta = _clamp(ability.theta, -self._theta_clip, self._theta_clip)

        env_state.count += 1
        difficulty_step = self._difficulty_lr / math.sqrt(max(1, env_state.count))
        env_state.difficulty -= difficulty_step * env_state.discrimination * residual
        env_state.difficulty = _clamp(env_state.difficulty, -self._diff_clip, self._diff_clip)

        discr_step = self._discrimination_lr / math.sqrt(max(1, env_state.count))
        env_state.discrimination += discr_step * (ability.theta - bias) * residual
        env_state.discrimination = _clamp(
            env_state.discrimination,
            self._disc_bounds[0],
            self._disc_bounds[1],
        )

        if challenge_state is not None:
            challenge_state.count += 1
            chal_step = self._challenge_lr / math.sqrt(max(1, challenge_state.count))
            challenge_state.bias += chal_step * residual
            challenge_state.bias = _clamp(challenge_state.bias, -self._diff_clip, self._diff_clip)

        return prob

    def predict(self, *, miner_key: Tuple[str, str], env: str, challenge_id: Optional[str] = None) -> float:
        ability = self._ensure_ability(miner_key)
        env_state = self._ensure_env(env)
        bias = env_state.difficulty
        if challenge_id:
            chal = self._challenge_state.get(challenge_id)
            if chal is not None:
                bias += chal.bias
        eta = env_state.discrimination * (ability.theta - bias)
        return _sigmoid(eta)

    def miner_params(self, miner_key: Tuple[str, str]) -> Tuple[float, int]:
        ability = self._ensure_ability(miner_key)
        return ability.theta, ability.count

    def env_params(self, env: str) -> Tuple[float, float]:
        env_state = self._ensure_env(env)
        return env_state.discrimination, env_state.difficulty

    def challenge_params(self, challenge_id: str) -> Tuple[float, int]:
        chal = self._challenge_state.get(challenge_id)
        if chal is None:
            return 0.0, 0
        return chal.bias, chal.count


# Central env registry
from .envs import ENVS

# --------------------------------------------------------------------------- #
#                   S3 helpers                                                #
# --------------------------------------------------------------------------- #
# ── ENV ──────────────────────────────────────────────────────────────────────
WINDOW        = int(os.getenv("AFFINE_WINDOW", 20))
RESULT_PREFIX = "affine/results/"
INDEX_KEY     = "affine/index.json"

FOLDER  = os.getenv("R2_FOLDER", "affine" )
BUCKET  = os.getenv("R2_BUCKET_ID", "80f15715bb0b882c9e967c13e677ed7d" )
ACCESS  = os.getenv("R2_WRITE_ACCESS_KEY_ID", "ff3f4f078019b064bfb6347c270bee4d")
SECRET  = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "a94b20516013519b2959cbbb441b9d1ec8511dce3c248223d947be8e85ec754d")
REGION  = os.getenv("R2_REGION", "auto")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

_S3_SIGNATURE = os.getenv("R2_SIGNATURE_VERSION", "s3v4")
_MAX_POOL = int(os.getenv("R2_MAX_POOL_CONNECTIONS", "256"))
S3_CLIENT_CONFIG = Config(
    signature_version=_S3_SIGNATURE,
    max_pool_connections=_MAX_POOL,
)

def get_client_ctx():
    """Return an aiobotocore S3 client configured for Cloudflare R2."""

    return get_session().create_client(
        "s3",
        region_name=REGION,
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS,
        aws_secret_access_key=SECRET,
        config=S3_CLIENT_CONFIG,
    )

CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR",
                 Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _w(b: int) -> int: return (b // WINDOW) * WINDOW

# ── fast JSON ───────────────────────────────────────────────────────────────
try:
    import orjson as _json
    _loads, _dumps = _json.loads, _json.dumps
except ModuleNotFoundError:
    _loads = lambda b: json.loads(b.decode())
    _dumps = lambda o: json.dumps(o, separators=(",", ":")).encode()
    
# ── Index helpers ───────────────────────────────────────────────────────────
async def _index() -> list[str]:
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
        except c.exceptions.NoSuchKey:
            logger.info("R2 index %s missing; attempting prefix scan under %s", INDEX_KEY, RESULT_PREFIX)
            paginator = c.get_paginator("list_objects_v2")
            keys: list[str] = []
            prefix_candidates = [RESULT_PREFIX]
            if RESULT_PREFIX and not RESULT_PREFIX.startswith("results"):
                prefix_candidates.append(RESULT_PREFIX.lstrip("/"))
            prefix_candidates.append("")  # fallback to whole bucket

            for prefix in prefix_candidates:
                paginate_kwargs = {"Bucket": FOLDER}
                if prefix:
                    paginate_kwargs["Prefix"] = prefix
                async for page in paginator.paginate(**paginate_kwargs):
                    for item in page.get("Contents", []):
                        key = item.get("Key")
                        if isinstance(key, str):
                            keys.append(key)
                if keys:
                    logger.info("Found %d R2 objects using prefix '%s'", len(keys), prefix)
                    break

            if not keys:
                raise RuntimeError(
                    "No R2 result shards found; index missing and prefix scan returned no objects. "
                    "Verify R2 bucket (%s) contains validator results." % FOLDER
                )

            return sorted(keys)

        payload = await r["Body"].read()
        data = json.loads(payload)
        if isinstance(data, dict) and "files" in data:
            files = data.get("files") or []
            return [f.get("key") for f in files if isinstance(f, dict) and f.get("key")]
        if isinstance(data, list):
            return [str(item) for item in data]
        raise RuntimeError(f"Unsupported index payload format for {INDEX_KEY}: {type(data).__name__}")

async def _update_index(k: str) -> None:
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
            idx = set(json.loads(await r["Body"].read()))
        except c.exceptions.NoSuchKey:
            idx = set()
        if k not in idx:
            idx.add(k)
            await c.put_object(Bucket=FOLDER, Key=INDEX_KEY,
                               Body=_dumps(sorted(idx)),
                               ContentType="application/json")

# ── Shard cache ─────────────────────────────────────────────────────────────
async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    name, out = Path(key).name, None
    out = CACHE_DIR / f"{name}.jsonl"; mod = out.with_suffix(".modified")
    async with sem, get_client_ctx() as c:
        if out.exists() and mod.exists():
            h = await c.head_object(Bucket=FOLDER, Key=key)
            if h["LastModified"].isoformat() == mod.read_text().strip():
                return out
        o = await c.get_object(Bucket=FOLDER, Key=key)
        body, lm = await o["Body"].read(), o["LastModified"].isoformat()
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(b"\n".join(_dumps(i) for i in _loads(body)) + b"\n")
    os.replace(tmp, out); mod.write_text(lm)
    return out

# ── Local JSON‑Lines iterator ───────────────────────────────────────────────
async def _jsonl(p: Path):
    try:
        import aiofiles
        async with aiofiles.open(p, "rb") as f:
            async for l in f: yield l.rstrip(b"\n")
    except ModuleNotFoundError:
        def _read():                         # run in thread
            with p.open("rb") as f: return f.read().splitlines()
        for l in await asyncio.to_thread(_read): yield l

# ── Core async stream (Result objects) ──────────────────────────────────────
async def dataset(
    tail: int,
    *,
    max_concurrency: int = 10,      # parallel S3 downloads
) -> AsyncIterator["Result"]:
    """
    Stream `Result`s in deterministic order while pre‑downloading future
    shards concurrently.
    """
    # ── figure out which windows we need ────────────────────────────────
    sub  = await get_subtensor()
    cur  = await sub.get_current_block()
    need = {w for w in range(_w(cur - tail), _w(cur) + WINDOW, WINDOW)}
    keys = [
        k for k in await _index()
        if (h := Path(k).name.split("-", 1)[0]).isdigit() and int(h) in need
    ]
    keys.sort()    
    # ── helpers ────────────────────────────────
    sem = asyncio.Semaphore(max_concurrency)     # throttle S3
    async def _prefetch(key: str) -> Path:       # just downloads / caches
        return await _cache_shard(key, sem)
    tasks: list[asyncio.Task[Path]] = [
        asyncio.create_task(_prefetch(k)) for k in keys[:max_concurrency]
    ]
    next_key = max_concurrency            
    bar = tqdm(f"Dataset=({cur}, {cur - tail})", unit="res", dynamic_ncols=True)
    # ── main loop: iterate over keys in order ───────────────────────────
    for i, key in enumerate(keys):
        path = await tasks[i]
        if next_key < len(keys):
            tasks.append(asyncio.create_task(_prefetch(keys[next_key])))
            next_key += 1
        async for raw in _jsonl(path):
            try:
                r = Result.model_validate(_loads(raw))
                if r.verify():
                    bar.update(1)
                    yield r
            except Exception:
                pass
    bar.close()


async def rollouts(
    tail: int,
    *,
    max_concurrency: int = 10,
) -> AsyncIterator["Result"]:
    """Alias for ``dataset`` to match the newer validator code path."""

    async for item in dataset(tail, max_concurrency=max_concurrency):
        yield item


# --------------------------------------------------------------------------- #
#                     R2 query helpers for validator tooling                  #
# --------------------------------------------------------------------------- #
_R2_SHARD_ROWS_CACHE: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
_R2_SHARD_LOCKS: Dict[str, asyncio.Lock] = {}
_R2_FETCH_SEMS: Dict[int, asyncio.Semaphore] = {}


def _pair_key(hotkey: Any, revision: Any) -> Optional[Tuple[str, str]]:
    """Normalise (hotkey, revision) pairs for comparisons."""

    if hotkey is None:
        return None
    hk = str(hotkey)
    rev = "" if revision is None else str(revision)
    return hk, rev


def _get_fetch_sem() -> asyncio.Semaphore:
    """Reuse a per-loop semaphore for throttling shard downloads."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    sem = _R2_FETCH_SEMS.get(key)
    if sem is None:
        max_conc = int(os.getenv("AFFINE_R2_FETCH_CONCURRENCY", "8"))
        sem = asyncio.Semaphore(max(1, max_conc))
        _R2_FETCH_SEMS[key] = sem
    return sem


async def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL cache shard into memory on a background thread."""

    def _read() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("rb") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    rows.append(_loads(raw))
        return rows

    return await asyncio.to_thread(_read)


async def _load_shard_rows(key: str) -> Tuple[dt.datetime, List[Dict[str, Any]]]:
    """Return cached rows and last-modified timestamp for an R2 shard key."""

    lock = _R2_SHARD_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _R2_SHARD_LOCKS[key] = lock

    async with lock:
        sem = _get_fetch_sem()
        path = await _cache_shard(key, sem)
        mod_path = path.with_suffix(".modified")
        mod_iso = mod_path.read_text().strip()
        cache_entry = _R2_SHARD_ROWS_CACHE.get(key)
        if cache_entry and cache_entry[0] == mod_iso:
            rows = cache_entry[1]
        else:
            rows = await _read_jsonl(path)
            _R2_SHARD_ROWS_CACHE[key] = (mod_iso, rows)
        timestamp = dt.datetime.fromisoformat(mod_iso)
        return timestamp, rows


def _normalize_r2_row(
    raw: Dict[str, Any],
    *,
    key: str,
    index: int,
    last_modified: dt.datetime,
) -> Optional[Dict[str, Any]]:
    """Project a raw R2 result payload into the database-like row shape."""

    challenge = raw.get("challenge") or {}
    evaluation = raw.get("evaluation") or {}
    response = raw.get("response") or {}
    miner = raw.get("miner") or {}

    env_name = challenge.get("env") or evaluation.get("env")
    if env_name is None:
        return None
    env_name = str(env_name)

    env_version = challenge.get("extra", {}).get("env_version")
    if env_version is None:
        env_cls = ENVS.get(env_name)
        env_version = getattr(env_cls, "__version__", None) if env_cls else None

    hotkey = miner.get("hotkey") or raw.get("hotkey")
    if hotkey is None:
        return None
    hotkey_str = str(hotkey)

    revision = miner.get("revision")
    revision_str = None if revision is None else str(revision)

    model = miner.get("model")
    uid = miner.get("uid")
    miner_block = miner.get("block")

    challenge_id = challenge.get("challenge_id")
    score_val = evaluation.get("score")
    try:
        score = float(score_val) if score_val is not None else None
    except (TypeError, ValueError):
        score = None

    success_val = response.get("success")
    success = None if success_val is None else bool(success_val)

    return {
        "env_name": env_name,
        "env_version": env_version,
        "uid": uid,
        "hotkey": hotkey_str,
        "model": model,
        "revision": revision_str,
        "challenge_id": challenge_id,
        "score": score,
        "success": success,
        "miner_block": miner_block,
        "signer_hotkey": raw.get("hotkey"),
        "result_version": raw.get("version"),
        "r2_key": key,
        "r2_last_modified": last_modified,
        "_order": (last_modified, index),
    }


async def _iter_r2_rows(
    *,
    pairs: set[Tuple[str, str]],
    env_name: Optional[str],
    env_version: Optional[str],
    order: Literal["asc", "desc"],
) -> AsyncIterator[Dict[str, Any]]:
    """Yield filtered R2 rows matching the supplied miner revisions."""

    if not pairs:
        return

    keys = sorted(await _index())
    key_iter: Iterable[str]
    if order == "desc":
        key_iter = reversed(keys)
    else:
        key_iter = keys

    for key in key_iter:
        last_modified, rows = await _load_shard_rows(key)
        idx_iter = range(len(rows)) if order == "asc" else range(len(rows) - 1, -1, -1)
        for idx in idx_iter:
            row = _normalize_r2_row(rows[idx], key=key, index=idx, last_modified=last_modified)
            if row is None:
                continue
            if env_name is not None and row["env_name"] != env_name:
                continue
            if env_version is not None and row.get("env_version") != env_version:
                continue
            rev_key = (row["hotkey"], "" if row["revision"] is None else str(row["revision"]))
            if rev_key not in pairs:
                continue
            yield row


def _normalise_pairs(pairs: Sequence[Tuple[str, Optional[str]]]) -> set[Tuple[str, str]]:
    """Convert user-supplied (hotkey, revision) tuples into canonical keys."""

    normalised: set[Tuple[str, str]] = set()
    for hotkey, revision in pairs:
        key = _pair_key(hotkey, revision)
        if key is not None:
            normalised.add(key)
    return normalised


# --------------------------------------------------------------------------- #
async def sign_results( wallet, results ):
    try:
        signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
        timeout = aiohttp.ClientTimeout(connect=2, total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payloads = [str(r.challenge) for r in results]
            resp = await session.post(f"{signer_url}/sign", json={"payloads": payloads})
            if resp.status == 200:
                data = await resp.json()
                sigs = data.get("signatures") or []
                hotkey = data.get("hotkey")
                for r, s in zip(results, sigs):
                    r.hotkey = hotkey
                    r.signature = s
    except Exception as e:
        logger.info(f"sink: signer unavailable, using local signing: {type(e).__name__}: {e}")
        hotkey = wallet.hotkey.ss58_address
        for r in results: 
            r.sign(wallet)
    finally:
        return hotkey, results

# ── Minimal sink / misc helpers (optional) ──────────────────────────────────
async def sink(wallet: bt.wallet, results: list["Result"], block: int = None):
    if not results: return
    if block is None:
        sub = await get_subtensor(); block = await sub.get_current_block()
    valid = [r for r in results if getattr(r.response, "success", False)]
    if not valid:
        return
    hotkey, signed = await sign_results( wallet, valid )
    key = f"{RESULT_PREFIX}{_w(block):09d}-{hotkey}.json"
    dumped = [ r.model_dump(mode="json") for r in signed ]
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=key)
            merged = json.loads(await r["Body"].read()) + dumped
        except c.exceptions.NoSuchKey:
            merged = dumped
        await c.put_object(Bucket=FOLDER, Key=key, Body=_dumps(merged),
                           ContentType="application/json")
    if len(merged) == len(dumped):              # shard was new
        await _update_index(key)

async def prune(tail: int):
    sub = await get_subtensor(); cur = await sub.get_current_block()
    for f in CACHE_DIR.glob("*.jsonl"):
        b = f.name.split("-", 1)[0]
        if b.isdigit() and int(b) < cur - tail:
            try: f.unlink()
            except OSError: pass

# --------------------------------------------------------------------------- #
#                               QUERY                                         #
# --------------------------------------------------------------------------- #
# Lazy-initialised semaphore and shared HTTP client
_HTTP_SEMS: Dict[int, asyncio.Semaphore] = {}
_CLIENTS: Dict[int, aiohttp.ClientSession] = {}

async def _get_sem() -> asyncio.Semaphore:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    sem = _HTTP_SEMS.get(key)
    if sem is None:
        sem = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "400")))
        _HTTP_SEMS[key] = sem
    return sem

async def _get_client() -> aiohttp.ClientSession:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    client = _CLIENTS.get(key)
    if client is None or client.closed:
        limit = int(os.getenv("AFFINE_HTTP_CONCURRENCY", "400"))  # raise this
        conn = aiohttp.TCPConnector(
            limit=limit,              # match or exceed your semaphore
            limit_per_host=0,         # don’t artificially throttle per host
            ttl_dns_cache=300,        # cache DNS results
            enable_cleanup_closed=True
        )
        client = aiohttp.ClientSession(
            connector=conn,
            timeout=aiohttp.ClientTimeout(total=None)
        )
        _CLIENTS[key] = client
    return client


TERMINAL = {400, 404, 410}
async def query(prompt, model: str = "unsloth/gemma-3-12b-it", slug: str = "llm", timeout=150, retries=0, backoff=1) -> Response:
    url = f"https://{slug}.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
    start = time.monotonic()
    QCOUNT.labels(model=model).inc()
    R = lambda resp, at, err, ok: Response(response=resp, latency_seconds=time.monotonic()-start,
                                          attempts=at, model=model, error=err, success=ok)
    sess = await _get_client()
    sem = await _get_sem()
    for attempt in range(1, retries+2):
        try:
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            async with sem, sess.post(url, json=payload,
                                      headers=hdr, timeout=timeout) as r:
                    txt = await r.text(errors="ignore")
                    if r.status in TERMINAL: return R(None, attempt, f"{r.status}:{txt}", False)
                    r.raise_for_status()
                    content = (await r.json())["choices"][0]["message"]["content"]
                    return R(content, attempt, None, True)
        except Exception as e:
            if attempt > retries: return R(None, attempt, str(e), False)
            await asyncio.sleep(backoff * 2**(attempt-1) * (1 + random.uniform(-0.1, 0.1)))

LOG_TEMPLATE = (
    "[RESULT] "
    "{pct:>3.0f}% | "
    "U{uid:>3d} │ "
    "{model:<50s} │ "
    "{env:<3} │ "
    "{success:^4s} │ "
    "{score:>6.4f} │ "
    "{latency:>6.3f}s"
)
async def run(challenges, miners, timeout=240, retries=0, backoff=1 )-> List[Result]:
    if not isinstance(challenges, list): challenges = [challenges]
    if isinstance(miners, Miner): miners = [miners]
    if isinstance(miners, dict):  mmap = miners
    elif isinstance(miners, list) and all(hasattr(m, "uid") for m in miners):  mmap = {m.uid: m for m in miners}
    else: mmap = await miners(miners)
    
    logger.trace("Running challenges: %s on miners: %s", [chal.prompt[:30] for chal in challenges], list(mmap.keys()))
    response = []
    
    async def proc(miner, chal):
        resp = await query(chal.prompt, miner.model, miner.slug, timeout, retries, backoff)
        try: ev = await chal.evaluate(resp)
        except Exception as e: ev = Evaluation(env=chal.env, score=0.0, extra={"error": str(e), "evaluation_failed": True})
        return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)
    
    tasks = [ asyncio.create_task(proc(m, chal)) for m in mmap.values() if m.model for chal in challenges]  
    total = len(tasks); completed = 0
    for task in asyncio.as_completed(tasks): 
        result: Result = await task
        response.append(result); completed += 1
        logger.debug(
            LOG_TEMPLATE.format(
                pct    = completed / total * 100,
                env    = result.challenge.env.name,                   
                uid    = result.miner.uid,                 
                model  = result.miner.model[:50] or "",         
                success= "RECV" if result.response.success else "NULL",
                score  = result.evaluation.score,
                latency= result.response.latency_seconds
            )
        )
    return response


# --------------------------------------------------------------------------- #
#                              Miners                                         #
# --------------------------------------------------------------------------- #
async def get_chute(chutes_id: str) -> Dict[str, Any]:
    url = f"https://api.chutes.ai/chutes/{chutes_id}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    sess = await _get_client()
    async with sess.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as r:
        text = await r.text(errors="ignore")
        if r.status != 200:
            return None
        info = await r.json()
        for k in ('readme','cords','tagline','instances'):
            info.pop(k, None)
        info.get('image', {}).pop('readme', None)
        return info
        
async def get_chute_code(identifier: str) -> Optional[str]:
    url = f"https://api.chutes.ai/chutes/code/{identifier}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            return await r.text(errors="ignore")

async def get_latest_chute_id(model_name: str, api_key: Optional[str] = None) -> Optional[str]:
    token = api_key or os.getenv("CHUTES_API_KEY", ""); 
    if not token: return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.chutes.ai/chutes/", headers={"Authorization": token}) as r:
                if r.status != 200: return None
                data = await r.json()
    except Exception: return None
    chutes = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(chutes, list): return None
    for chute in reversed(chutes):
        if any(chute.get(k) == model_name for k in ("model_name", "name", "readme")):
            return chute.get("chute_id")
    return None


async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: object = None,
) -> Dict[int, Miner]:
    sub = await get_subtensor()
    meta = meta or await sub.metagraph(netuid)
    commits = await sub.get_all_revealed_commitments(netuid)
    if uids is None:uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int): uids = [uids]    
    meta_sem = asyncio.Semaphore(int(os.getenv("AFFINE_META_CONCURRENCY", "64")))
    async def fetch(uid: int):
        try:
            hotkey = meta.hotkeys[ uid ]
            if hotkey not in commits: return None
            commit = commits[hotkey]
            block, data = commit[-1]     
            block = 0 if uid == 0 else block
            data = json.loads(data)
            model, miner_revision, chute_id = data.get("model"), data.get("revision"), data.get("chute_id")
            async with meta_sem:
                chute = await get_chute(chute_id)
            if not chute: return None
            if not chute.get("hot", False): return None
            gated = await check_model_gated(model)
            if gated is None or gated is True: return None
            chutes_name, slug, chutes_revision = chute.get('name'), chute.get("slug"), chute.get("revision")
            if model != chutes_name or (uid != 0 and chutes_name.split('/')[1].lower()[:6] != 'affine'): return None
            if chutes_revision == None or miner_revision == chutes_revision:
                miner = Miner(
                    uid=uid, hotkey=hotkey, model=model, block=int(block),
                    revision = miner_revision,
                    slug = slug,
                    chute=chute,
                )
                return miner
        except: pass
    results = await asyncio.gather(*(fetch(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}
    # Remove duplicates.
    if output:
        best_by_model: Dict[str, Tuple[int, int]] = {}
        for uid, m in output.items():
            if not m.model:
                continue
            blk = m.block if isinstance(m.block, int) else (int(m.block) if m.block is not None else (2**63 - 1))
            prev = best_by_model.get(m.model)
            if prev is None or blk < prev[0]:
                best_by_model[m.model] = (blk, uid)
        selected_uids = {uid for _, uid in best_by_model.values()}
        output = {uid: m for uid, m in output.items() if uid in selected_uids}
    return output


async def get_miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: object = None,
) -> Dict[int, Miner]:
    """Alias retained for backwards compatibility with the IRT validator code."""
    return await miners(uids=uids, netuid=netuid, meta=meta)


async def fetch_recent_results(
    *,
    pairs: List[Tuple[str, str]],
    env_name: Optional[str] = None,
    env_version: Optional[str] = None,
    limit: int = 1000,
    ascending: bool = True,
) -> List[Dict[str, Any]]:
    """Fetch recent validator results directly from R2 storage."""

    if not pairs or limit <= 0:
        return []

    pair_keys = _normalise_pairs(pairs)
    if not pair_keys:
        return []

    collected: List[Dict[str, Any]] = []
    async for row in _iter_r2_rows(
        pairs=pair_keys,
        env_name=env_name,
        env_version=env_version,
        order="desc",
    ):
        collected.append(row)
        if len(collected) >= limit:
            break

    if not collected:
        return []

    collected.sort(key=lambda r: r["_order"], reverse=not ascending)
    for row in collected:
        row.pop("_order", None)
    return collected


async def aggregate_success_by_env(
    *,
    env_name: str,
    pairs: List[Tuple[str, str]],
    env_version: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    if not pairs:
        return {}

    pair_keys = _normalise_pairs(pairs)
    if not pair_keys:
        return {}

    totals: Dict[str, Dict[str, float]] = {}
    async for row in _iter_r2_rows(
        pairs=pair_keys,
        env_name=env_name,
        env_version=env_version,
        order="asc",
    ):
        if row.get("success") is True:
            hotkey = row["hotkey"]
            stats = totals.setdefault(hotkey, {"n_success": 0.0, "sum_score": 0.0})
            stats["n_success"] += 1.0
            score = row.get("score")
            if score is not None:
                stats["sum_score"] += float(score)
    return totals


async def aggregate_scores_by_env(
    *,
    env_name: str,
    pairs: List[Tuple[str, str]],
    env_version: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    if not pairs:
        return {}

    pair_keys = _normalise_pairs(pairs)
    if not pair_keys:
        return {}

    totals: Dict[str, Dict[str, float]] = {}
    async for row in _iter_r2_rows(
        pairs=pair_keys,
        env_name=env_name,
        env_version=env_version,
        order="asc",
    ):
        score = row.get("score")
        if score is None:
            continue
        hotkey = row["hotkey"]
        stats = totals.setdefault(hotkey, {"n_total": 0.0, "sum_score": 0.0, "sum_sq_score": 0.0})
        score_f = float(score)
        stats["n_total"] += 1.0
        stats["sum_score"] += score_f
        stats["sum_sq_score"] += score_f * score_f
    return totals


# --------------------------------------------------------------------------- #
#                         Validator scoring pipeline                          #
# --------------------------------------------------------------------------- #


def _mk_key(hk: Union[str, Any], rev: Optional[str]) -> Tuple[str, str]:
    return (str(hk), str(rev) if rev is not None else "")


def _ingest_observations(
    irt: OnlineIRT2PL,
    observations: Iterable[Dict[str, Any]],
    *,
    pair_keys: Collection[Tuple[str, str]],
) -> Tuple[int, DefaultDict[Tuple[str, str], DefaultDict[str, int]]]:
    counts_pair: DefaultDict[Tuple[str, str], DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_rows = 0
    for obs in observations:
        miner_key = _mk_key(obs["hotkey"], obs["revision"])
        if miner_key not in pair_keys:
            continue
        success_flag = obs["success"]
        if success_flag is None:
            success_flag = 1.0 if (obs.get("score") or 0.0) > 0.0 else 0.0
        else:
            success_flag = 1.0 if success_flag else 0.0
        env = obs["env"]
        irt.observe(
            miner_key=miner_key,
            env=env,
            success=success_flag,
            challenge_id=obs.get("challenge_id"),
        )
        counts_pair[miner_key][env] += 1
        total_rows += 1
    return total_rows, counts_pair


def _compute_predictions(
    *,
    irt: OnlineIRT2PL,
    env_names: Sequence[str],
    meta_hotkeys: Sequence[str],
    key_by_hotkey: Mapping[str, Tuple[str, str]],
    counts_pair: Mapping[Tuple[str, str], Mapping[str, int]],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    cnt: Dict[str, Dict[str, int]] = {}
    mean: Dict[str, Dict[str, float]] = {}
    variance: Dict[str, Dict[str, float]] = {}
    for hk in meta_hotkeys:
        pair_key = key_by_hotkey.get(hk)
        cnt_row: Dict[str, int] = {}
        mean_row: Dict[str, float] = {}
        var_row: Dict[str, float] = {}
        for env in env_names:
            n_obs = 0
            p = 0.5
            if pair_key is not None:
                env_counts = counts_pair.get(pair_key)
                if env_counts is not None:
                    n_obs = int(env_counts.get(env, 0))
                p = irt.predict(miner_key=pair_key, env=env)
            cnt_row[env] = n_obs
            mean_row[env] = p
            var_row[env] = max(0.0, p * (1.0 - p))
        cnt[hk] = cnt_row
        mean[hk] = mean_row
        variance[hk] = var_row
    return cnt, mean, variance


TAIL = 20_000
EPS_FLOOR = 0.005
Z_NOT_WORSE = 1.28
EPS_WIN = 0.008
Z_WIN = 0.5
ELIG = 0.03


async def get_weights(tail: int = TAIL, scale: float = 1.0):
    """Compute validator weights using ε-Pareto dominance over env subsets."""

    subtensor = await get_subtensor()
    meta = await subtensor.metagraph(NETUID)
    env_names = tuple(str(e) for e in ENVS)
    n_envs = len(env_names)

    current_miners = await get_miners(meta=meta)
    prev = {m.hotkey: m for m in current_miners.values()}
    first_block = {m.hotkey: m.block for m in current_miners.values()}
    pairs_db = [(mi.hotkey, mi.revision) for mi in current_miners.values()]
    pair_keys = {_mk_key(hk, rev) for hk, rev in pairs_db}
    key_by_hotkey = {m.hotkey: _mk_key(m.hotkey, m.revision) for m in current_miners.values()}

    irt = OnlineIRT2PL(env_names=env_names)
    observations: List[Dict[str, Any]] = []

    for env_name in env_names:
        env_cls = ENVS.get(env_name)
        env_version = getattr(env_cls, "__version__", None)
        try:
            rows = await fetch_recent_results(
                pairs=pairs_db,
                env_name=env_name,
                env_version=env_version,
                limit=tail,
                ascending=True,
            )
            for row in rows:
                observations.append(
                    {
                        "env": str(row.get("env_name", env_name)),
                        "hotkey": row.get("hotkey"),
                        "revision": row.get("revision"),
                        "success": row.get("success"),
                        "score": row.get("score"),
                        "challenge_id": row.get("challenge_id"),
                        "ts": row.get("r2_last_modified"),
                    }
                )
        except Exception as exc:
            logger.warning("Error pulling recent results for env %s: %s", env_name, exc)

    observations.sort(key=lambda r: r["ts"] or dt.datetime.min)
    total_rows, counts_pair = _ingest_observations(irt, observations, pair_keys=pair_keys)
    cnt, mean, variance = _compute_predictions(
        irt=irt,
        env_names=env_names,
        meta_hotkeys=meta.hotkeys,
        key_by_hotkey=key_by_hotkey,
        counts_pair=counts_pair,
    )

    active_hks = list(prev.keys())
    for env in env_names:
        max_e = max((mean.get(hk, {}).get(env, 0.0) for hk in active_hks), default=0.0)
        MAXENV.labels(env=env).set(max_e)
        a_env, b_env = irt.env_params(env)
        logger.debug("[IRT] env=%s a=%.3f b=%.3f", env, a_env, b_env)

    logger.info("Computed online 2PL IRT & updated MAXENV (rows=%d, miners=%d).", total_rows, len(active_hks))

    required = {
        e: 10 + int(ELIG * max((cnt[hk][e] for hk in active_hks), default=0))
        for e in env_names
    }
    eligible = {hk for hk in active_hks if all(cnt[hk][e] >= required[e] for e in env_names)}

    def _var(hk: str, e: str) -> float:
        n = cnt[hk][e]
        if n <= 1:
            return 0.0
        return variance[hk][e]

    def thr_not_worse(a_i: float, n_i: int, v_i: float, a_j: float, n_j: int, v_j: float) -> float:
        if Z_NOT_WORSE <= 0:
            return EPS_FLOOR
        se = math.sqrt((v_i / max(n_i, 1)) + (v_j / max(n_j, 1)))
        return max(EPS_FLOOR, Z_NOT_WORSE * se)

    def thr_better(a_i: float, n_i: int, v_i: float, a_j: float, n_j: int, v_j: float, nw: float) -> float:
        if Z_WIN > 0:
            se = math.sqrt((v_i / max(n_i, 1)) + (v_j / max(n_j, 1)))
            t = max(EPS_WIN, Z_WIN * se)
        else:
            t = EPS_WIN
        return min(t, nw)

    def dominates_on(a: str, b: str, subset: Sequence[str]) -> bool:
        not_worse_all = True
        better_any = False
        tie_all = True
        for e in subset:
            ai, aj = mean[a][e], mean[b][e]
            ni, nj = cnt[a][e], cnt[b][e]
            vi, vj = _var(a, e), _var(b, e)
            nw = thr_not_worse(ai, ni, vi, aj, nj, vj)
            bet = thr_better(ai, ni, vi, aj, nj, vj, nw)

            if ai < aj - nw:
                not_worse_all = False
            if ai >= aj + bet:
                better_any = True
            if abs(ai - aj) > nw:
                tie_all = False

        if not_worse_all and better_any:
            return True
        if tie_all and first_block.get(a, float("inf")) < first_block.get(b, float("inf")):
            return True
        return False

    dom_full = defaultdict(int)
    pool_for_dom = eligible if eligible else set(active_hks)
    for a, b in itertools.permutations(pool_for_dom, 2):
        if dominates_on(a, b, env_names):
            dom_full[a] += 1
    logger.info("Computed ε-dominance counts (full env set).")

    def ts(hk: str) -> int:
        return int(first_block.get(hk, prev[hk].block))

    best = max(pool_for_dom, key=lambda hk: (dom_full.get(hk, 0), -ts(hk))) if pool_for_dom else active_hks[0]
    best_uid = meta.hotkeys.index(best)

    def layer_weights(n: int, kappa: float):
        weights = {1: kappa}
        for s in range(2, n + 1):
            weights[s] = kappa * (2**s)
        return weights

    def subset_winner(env_subset: Sequence[str]):
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool_for_dom, 2):
            if dominates_on(x, y, env_subset):
                dom_local[x] += 1
        return max(pool_for_dom, key=lambda hk: (dom_local.get(hk, 0), -ts(hk)))

    layer_points = {hk: defaultdict(float) for hk in active_hks}
    score = defaultdict(float)
    env_winners = {e: subset_winner((e,)) for e in env_names}

    k_by_layer = layer_weights(n_envs, scale)
    for s in range(1, n_envs + 1):
        for env_subset in itertools.combinations(env_names, s):
            winner = subset_winner(env_subset)
            score[winner] += k_by_layer[s]
            layer_points[winner][s] += k_by_layer[s]

    if not eligible:
        logger.warning("No eligible miners; assigning weight 1.0 to canonical best.")
        for uid, hk in enumerate(meta.hotkeys):
            WEIGHT.labels(uid=uid).set(1.0 if hk == best else 0.0)
            for env in env_names:
                a = mean[hk][env]
                if cnt[hk][env] > 0:
                    SCORE.labels(uid=uid, env=env).set(a)

        hdr = (
            ["UID", "Model", "Rev"]
            + [f"{e}" for e in env_names]
            + [f"L{s}" for s in range(1, n_envs + 1)]
            + ["Pts", "Elig", "Wgt"]
        )

        def row(hk: str):
            m = prev[hk]
            w = 1.0 if hk == best else 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for env in env_names:
                base = f"{100 * mean[hk][env]:.2f}/{cnt[hk][env]}"
                env_cols.append(f"*{base}*" if hk == env_winners.get(env) else base)
            return [
                m.uid,
                model_name,
                str(m.revision)[:5],
                *env_cols,
                *[f"{layer_points[hk][s]:.1f}" for s in range(1, n_envs + 1)],
                f"{score.get(hk, 0.0):.2f}",
                "Y" if hk in eligible else "N",
                f"{w:.4f}",
            ]

        rows = sorted((row(hk) for hk in active_hks), key=lambda r: (r[-3], r[0]), reverse=True)
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))
        return [best_uid], [1.0]

    total_points = sum(score[hk] for hk in eligible)
    if total_points <= 0:
        logger.warning("Combinatoric scoring returned zero total; falling back to canonical best.")
        weight_by_hk = {hk: (1.0 if hk == best else 0.0) for hk in eligible}
    else:
        weight_by_hk = {hk: (score[hk] / total_points) for hk in eligible}

    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in env_names]
        + [f"L{s}" for s in range(1, n_envs + 1)]
        + ["Pts", "Elig", "Wgt"]
    )

    def row(hk: str):
        m = prev[hk]
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for env in env_names:
            base = f"{100 * mean[hk][env]:.2f}/{cnt[hk][env]}"
            env_cols.append(f"*{base}*" if hk == env_winners.get(env) else base)
        return [
            m.uid,
            model_name[:30],
            str(m.revision)[:5],
            *env_cols,
            *[f"{layer_points[hk][s]:.1f}" for s in range(1, n_envs + 1)],
            f"{score.get(hk, 0.0):.2f}",
            "Y" if hk in eligible else "N",
            f"{w:.4f}",
        ]

    ranked_rows = sorted((row(hk) for hk in eligible), key=lambda r: float(r[-3]), reverse=True)
    unranked_rows = sorted((row(hk) for hk in active_hks if hk not in eligible), key=lambda r: float(r[-3]), reverse=True)
    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    for uid, hk in enumerate(meta.hotkeys):
        WEIGHT.labels(uid=uid).set(weight_by_hk.get(hk, 0.0))
        for env in env_names:
            a = mean[hk][env]
            if cnt[hk][env] > 0:
                SCORE.labels(uid=uid, env=env).set(a)

    eligible_uids = [meta.hotkeys.index(hk) for hk in eligible]
    uids = [u for u in eligible_uids if u != best_uid] + [best_uid]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights


def validate():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    async def _run():
        LAST = 0
        TEMPO = 360
        INNER_TEMPO = 100
        subtensor = None
        while True:
            try:
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
                if subtensor is None:
                    subtensor = await get_subtensor()
                block = await subtensor.get_current_block()
                interval = (TEMPO + 1 + NETUID + 1 + block) % (TEMPO + 1) % INNER_TEMPO
                if interval != 0:
                    logger.debug(
                        "Waiting ... (%s + 1 + %s + 1 + %s) %% (%s + 1) %% %s = %s != 0",
                        TEMPO,
                        NETUID,
                        block,
                        TEMPO,
                        INNER_TEMPO,
                        interval,
                    )
                    await subtensor.wait_for_block()
                    continue

                uids, weights = await get_weights()

                logger.info("Setting weights ...")
                await retry_set_weights(wallet, uids=uids, weights=weights, retry=3)
                subtensor = await get_subtensor()
                set_block = await subtensor.get_current_block()
                LASTSET.set_function(lambda: set_block - LAST)
                LAST = block

            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - operational loop
                traceback.print_exc()
                logger.info("Error in validator loop: %s. Continuing ...", exc)
                subtensor = None
                await asyncio.sleep(10)
                continue

    async def _main():
        await asyncio.gather(_run(), watchdog(timeout=60 * 20))

    asyncio.run(_main())
def weights_cmd():
    asyncio.run(get_weights())


validator = SimpleNamespace(
    OnlineIRT2PL=OnlineIRT2PL,
    _mk_key=_mk_key,
    _ingest_observations=_ingest_observations,
    _compute_predictions=_compute_predictions,
    get_weights=get_weights,
    validate=validate,
    weights=weights_cmd,
)


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)
    
cli.add_command(click.command("validate")(validate))
cli.add_command(click.command("weights")(weights_cmd))

# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = None

async def watchdog(timeout: int = 600, sleep_div: float = 6.0):
    sleep = timeout / sleep_div
    while HEARTBEAT is None:
        await asyncio.sleep(sleep)
    while True:
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)
        await asyncio.sleep(sleep)
            
# --------------------------------------------------------------------------- #
#                               Runner                                        #
# --------------------------------------------------------------------------- #
import contextlib
@cli.command("runner")
def runner():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT",  "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)

    async def _run():
        subtensor = None
        envs = [cls() for cls in ENVS.values()]

        # ── config ───────────────────────────────────────────────────────────
        MAX_USES       = 30
        REFRESH_S      = 600     # metagraph/miners refresh cadence (s)
        SINK_BATCH     = 300     # flush threshold
        SINK_MAX_WAIT  = 60*5      # max seconds to hold partial batch
        BACKOFF0       = 5
        BACKOFF_CAP    = 300

        # ── state ───────────────────────────────────────────────────────────
        chal_cache, i_env = {}, 0
        last_sync = 0.0
        delay = defaultdict(lambda: BACKOFF0)   # uid -> current delay
        cooldown_until = defaultdict(float)     # uid -> t when allowed again
        miners_map = {}

        # results pipeline
        sink_q: asyncio.Queue = asyncio.Queue()

        # monitoring state
        last_status_log = 0.0
        total_requests = 0
        requests_since_last_log = 0

        def ok(res_list):
            if not res_list: return False
            r = res_list[0]
            if not r.response.success: return False
            return True

        async def next_chal():
            nonlocal i_env
            e = envs[i_env]; i_env = (i_env + 1) % len(envs)
            chal, uses = chal_cache.get(e, (None, 0))
            if chal is None or uses >= MAX_USES:
                chal, uses = await e.generate(), 0
            chal_cache[e] = (chal, uses + 1)
            return chal

        async def schedule(miner, inflight, now):
            nonlocal total_requests, requests_since_last_log
            uid = int(miner.uid)
            if uid in inflight: return
            if now < cooldown_until[uid]: return
            chal = await next_chal()
            inflight[uid] = asyncio.create_task(run(chal, miner, timeout=180))
            total_requests += 1
            requests_since_last_log += 1

        async def ensure_subtensor():
            nonlocal subtensor
            if subtensor is None:
                subtensor = await get_subtensor()
            return subtensor

        async def refresh_miners(now):
            nonlocal last_sync, miners_map
            if (now - last_sync) >= REFRESH_S or last_sync == 0:
                st = await ensure_subtensor()
                meta = await st.metagraph(NETUID)
                miners_map = await miners(meta=meta)
                last_sync = now
                logger.debug(f"refresh: miners={len(miners_map)}")

        async def sink_worker():
            """Consumes results from sink_q and flushes in batches of SINK_BATCH or after SINK_MAX_WAIT."""
            nonlocal subtensor
            batch = []
            first_put_time = None
            while True:
                try:
                    # If we have started a batch, only wait up to the remaining hold time; otherwise wait for first item.
                    if first_put_time is None:
                        logger.debug(f"sink_worker: queue size={sink_q.qsize()}")
                        item = await sink_q.get()
                        first_put_time = time.monotonic()
                        batch.append(item)
                        # Opportunistically drain without blocking to build the batch quickly
                        while len(batch) < SINK_BATCH:
                            try:
                                more = sink_q.get_nowait()
                                batch.append(more)
                            except asyncio.QueueEmpty:
                                break
                    else:
                        remaining = SINK_MAX_WAIT - (time.monotonic() - first_put_time)
                        timeout = remaining if remaining > 0.05 else 0.05
                        try:
                            item = await asyncio.wait_for(sink_q.get(), timeout=timeout)
                            batch.append(item)
                            while len(batch) < SINK_BATCH:
                                try:
                                    more = sink_q.get_nowait()
                                    batch.append(more)
                                except asyncio.QueueEmpty:
                                    break
                        except asyncio.TimeoutError:
                            pass

                    elapsed = (time.monotonic() - first_put_time) if first_put_time is not None else 0.0
                    logger.debug(f"Until Sink: {len(batch)}/{SINK_BATCH} Time: {elapsed}/{SINK_MAX_WAIT}")
                    await asyncio.sleep(3)
                    if len(batch) >= SINK_BATCH or (batch and elapsed >= SINK_MAX_WAIT):
                        st = await ensure_subtensor()
                        blk = await st.get_current_block()
                        # Flatten: items may be single Result or list[Result]
                        flat = []
                        for it in batch:
                            if isinstance(it, list):
                                flat.extend(it)
                            else:
                                flat.append(it)
                        logger.debug(f"sink_worker: flushing {len(flat)} results")
                        try:
                            await sink(wallet=wallet, block=blk, results=flat)
                        except Exception:
                            traceback.print_exc()
                            # keep going; don't drop future batches
                        batch.clear()
                        first_put_time = None
                except asyncio.CancelledError:
                    # drain and final flush
                    flat = []
                    while not sink_q.empty():
                        it = sink_q.get_nowait()
                        if isinstance(it, list): flat.extend(it)
                        else: flat.append(it)
                    if flat:
                        try:
                            st = await ensure_subtensor()
                            blk = await st.get_current_block()
                            logger.debug(f"sink_worker: final flush {len(flat)}")
                            await sink(wallet=wallet, block=blk, results=flat)
                        except Exception:
                            traceback.print_exc()
                    break

        async def main_loop():
            global HEARTBEAT
            nonlocal last_status_log, requests_since_last_log
            inflight = {}
            sink_task = asyncio.create_task(sink_worker())
            try:
                while True:
                    HEARTBEAT = now = time.monotonic()
                    # heartbeat + ensure subtensor
                    _ = await ensure_subtensor()
                    # periodic refresh
                    await refresh_miners(now)
                    if not miners_map:
                        await asyncio.sleep(1)
                        continue

                    # periodic status logging
                    if now - last_status_log >= 30:
                        elapsed = now - last_status_log if last_status_log > 0 else 30
                        rps = requests_since_last_log / elapsed
                        cooldown_count = sum(1 for uid in miners_map.keys() if now < cooldown_until[uid])
                        queue_size = sink_q.qsize()
                        logger.info(f"[STATUS] miners={len(miners_map)} inflight={len(inflight)} cooldown={cooldown_count} queue={queue_size} req/s={rps:.1f} total_req={total_requests}")
                        last_status_log = now
                        requests_since_last_log = 0

                    # seed/respect cooldowns
                    for m in miners_map.values():
                        await schedule(m, inflight, now)

                    if not inflight:
                        await asyncio.sleep(0.2)
                        continue

                    done, _ = await asyncio.wait(inflight.values(), return_when=asyncio.FIRST_COMPLETED)
                    HEARTBEAT = now = time.monotonic()
                    for t in done:
                        uid = next((u for u, tk in list(inflight.items()) if tk is t), None)
                        miner = miners_map.get(uid)
                        inflight.pop(uid, None)
                        try:
                            res_list = await t  # list[Result]; may be []
                        except Exception as e:
                            logger.debug(f"miner {uid} task error: {e}")
                            res_list = []

                        if ok(res_list):
                            # reset backoff, enqueue results (non-blocking)
                            delay[uid] = BACKOFF0
                            cooldown_until[uid] = now
                            # push entire list; sink worker will flatten
                            sink_q.put_nowait(res_list)
                            queue_size = sink_q.qsize()
                            logger.debug(f"miner {uid} OK; queued {len(res_list)}, queue_size={queue_size}")
                        else:
                            print ('not ok')
                            # exponential backoff + jitter
                            d = min(delay[uid] * 2, BACKOFF_CAP)
                            jitter = random.uniform(0, d * 0.2)
                            delay[uid] = d
                            cooldown_until[uid] = now + d + jitter
                            logger.debug(f"miner {uid} FAIL; cooldown {d:+.1f}s(+{jitter:.1f})")

                        # try to reschedule
                        if miner:
                            await schedule(miner, inflight, now)
            except asyncio.CancelledError:
                pass
            finally:
                # cancel sink worker and wait for final flush
                sink_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sink_task

        await main_loop()

    async def main():
        await asyncio.gather(_run(), watchdog(timeout=600))

    asyncio.run(main())



# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #
    
async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "",
) -> bool:
    for attempt in range(retries):
        try:
            st = await get_subtensor()
            ref = await st.get_current_block()
            logger.info(f"{log_prefix} set_weights attempt {attempt+1}/{retries}: netuid={netuid} uids={uids} weights={weights}")
            start = time.monotonic()
            bt.subtensor(get_conf('SUBTENSOR_ENDPOINT', default='finney')).set_weights(
                wallet=wallet, netuid=netuid, weights=weights, uids=uids,
                wait_for_inclusion=wait_for_inclusion,
            )
            logger.info(f"{log_prefix} extrinsic submitted in {(time.monotonic()-start)*1000:.1f}ms; waiting next block … (ref_block={ref})")
            await st.wait_for_block()
            meta = await st.metagraph(netuid)
            try:
                idx = meta.hotkeys.index(wallet.hotkey.ss58_address)
                lu = meta.last_update[idx]
                logger.info(f"{log_prefix} last_update={lu}, ref_block={ref}")
                if lu >= ref:
                    logger.info(f"{log_prefix} confirmation OK (last_update >= ref)")
                    return True
                logger.warning(f"{log_prefix} confirmation not yet included (last_update < ref), retrying …")
            except ValueError:
                logger.warning(f"{log_prefix} wallet hotkey not found in metagraph hotkeys; retrying …")
        except Exception as e:
            logger.warning(f"{log_prefix} set_weights attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}")
        await asyncio.sleep(delay_s)
    return False

@cli.command("signer")
@click.option('--host', default=os.getenv('SIGNER_HOST', '0.0.0.0'))
@click.option('--port', default=int(os.getenv('SIGNER_PORT', '8080')))
def signer(host: str, port: int):
    """Start lightweight HTTP signer service."""
    async def _run():
        from aiohttp import web
        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)
        @web.middleware
        async def access_log(request: "web.Request", handler):
            start = time.monotonic()
            try:
                resp = await handler(request)
                return resp
            finally:
                dur = (time.monotonic() - start) * 1000
                logger.info(
                    f"[signer] {request.remote} {request.method} {request.path} -> {getattr(request, 'response', None) and getattr(request.response, 'status', '?')} {dur:.1f}ms"
                )

        async def health(_request: "web.Request"):
            return web.json_response({"ok": True})
    
        async def sign_handler(request: "web.Request"):
            try:
                payload = await request.json()
                data = payload.get("payloads") or payload.get("data") or []
                if isinstance(data, str):
                    data = [data]
                sigs = [(wallet.hotkey.sign(data=d)).hex() for d in data]
                return web.json_response({
                    "success": True,
                    "signatures": sigs,
                    "hotkey": wallet.hotkey.ss58_address
                })
            except Exception as e:
                logger.error(f"[signer] /sign error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)


        async def set_weights_handler(request: "web.Request"):
            try:
                logger.info("[signer] /set_weights: request received")
                payload = await request.json()
                netuid = int(payload.get('netuid', NETUID))
                uids = payload.get('uids') or []
                weights = payload.get('weights') or []
                wait_for_inclusion = bool(payload.get('wait_for_inclusion', False))
                ok = await _set_weights_with_confirmation(
                    wallet,
                    netuid,
                    uids,
                    weights,
                    wait_for_inclusion,
                    retries=int(os.getenv("SIGNER_RETRIES", "10")),
                    delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
                    log_prefix="[signer]",
                )
                logger.info(f"[signer] /set_weights: confirmation={'ok' if ok else 'failed'}")
                return web.json_response({"success": True} if ok else {"success": False, "error": "confirmation failed"}, status=200 if ok else 500)
            except Exception as e:
                logger.error(f"[signer] set_weights error: {e}")
                return web.json_response({"success": False, "error": str(e)}, status=500)

        app = web.Application(middlewares=[access_log])
        app.add_routes([
            web.get('/healthz', health),
            web.post('/set_weights', set_weights_handler),
            web.post('/sign', sign_handler),
        ])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
        try:
            hn = socket.gethostname()
            ip = socket.gethostbyname(hn)
        except Exception:
            hn, ip = ("?", "?")
        logger.info(f"Signer service listening on http://{host}:{port} hostname={hn} ip={ip}")
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_run())

async def retry_set_weights( wallet: bt.Wallet, uids: List[int], weights: List[float], retry: int = 10 ):
    # Delegate to signer; fallback to shared helper only if signer is unreachable
    signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
    try:
        logger.info(f"Calling signer at {signer_url} for set_weights uids={uids}")
        parsed = urlparse(signer_url)
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80, proto=socket.IPPROTO_TCP)
            addrs = ",".join(sorted({i[4][0] for i in infos}))
            logger.info(f"Signer DNS: host={parsed.hostname} -> {addrs}")
        except Exception as e:
            logger.warning(f"DNS resolve failed for {parsed.hostname}: {e}")
        timeout = aiohttp.ClientTimeout(connect=2, total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            start = time.monotonic()
            resp = await session.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": NETUID,
                    "weights": weights,
                    "uids": uids,
                    "wait_for_inclusion": False,
                },
            )
            dur_ms = (time.monotonic() - start) * 1000
            logger.info(f"Signer HTTP response status={resp.status} in {dur_ms:.1f}ms")
            # Try to parse JSON, otherwise log text (trimmed)
            try:
                data = await resp.json()
            except Exception:
                txt = await resp.text()
                data = {"raw": (txt[:500] + ('…' if len(txt) > 500 else ''))}
            logger.info(f"Signer response body={data}")
            if resp.status == 200 and data.get("success"):
                LASTSET.set(time.time())
                return
            # Do not fallback if signer exists but reports failure
            logger.warning(f"Signer responded error: status={resp.status} body={data}")
            return
    except ClientConnectorError as e:
        logger.info(f"Signer not reachable ({type(e).__name__}: {e}); falling back to local set_weights once")
        ok = await _set_weights_with_confirmation(
            wallet, NETUID, uids, weights, False,
            retries=int(os.getenv("SIGNER_RETRIES", "10")),
            delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
            log_prefix="[validator-fallback]",
        )
        if ok:
            LASTSET.set(time.time())
        else:
            logger.error("Local set_weights confirmation failed")
        return
    except asyncio.TimeoutError as e:
        logger.warning(f"Signer call timed out: {e}. Not falling back to local because validator has no wallet.")
        return
    
# --------------------------------------------------------------------------- #
#                              Pull Model                                     #
# --------------------------------------------------------------------------- #
@cli.command("pull")
@click.argument("uid", type=int)
@click.option("--model_path", "-p", default = './model_path', required=True, type=click.Path(), help="Local directory to save the model")
@click.option('--hf-token', default=None, help="Hugging Face API token (env HF_TOKEN if unset)")
def pull(uid: int, model_path: str, hf_token: str):
    """Pulls a model from a specific miner UID if exists."""

    # 1. Ensure HF token
    hf_token     = hf_token or get_conf("HF_TOKEN")

    # 2. Lookup miner on‑chain
    miner_map = asyncio.run(miners(uids=uid))
    miner = miner_map.get(uid)
    
    if miner is None:
        click.echo(f"No miner found for UID {uid}", err=True)
        sys.exit(1)
    repo_name = miner.model
    logger.info("Pulling model %s for UID %d into %s", repo_name, uid, model_path)

    # 3. Download snapshot
    try:
        snapshot_download(
            repo_id=repo_name,
            repo_type="model",
            local_dir=model_path,
            token=hf_token,
            resume_download=True,
            revision=miner.revision,
        )
        click.echo(f"Model {repo_name} pulled to {model_path}")
    except Exception as e:
        logger.error("Failed to download %s: %s", repo_name, e)
        click.echo(f"Error pulling model: {e}", err=True)
        sys.exit(1)


# --------------------------------------------------------------------------- #
#                              Push Model                                     #
# --------------------------------------------------------------------------- #
@cli.command("push")
@click.option('--model_path',  default='./model_path', help='Local path to model artifacts.')
@click.option('--existing-repo', default=None, help='Use an existing HF repo instead of uploading (format <user>/<repo>)')
@click.option('--revision', default=None, help='Commit SHA to register (only relevant with --existing-repo)')
@click.option('--coldkey',     default=None, help='Name of the cold wallet to use.')
@click.option('--hotkey',      default=None, help='Name of the hot wallet to use.')
@click.option('--chutes-api-key', default=None, help='Chutes API key (env CHUTES_API_KEY if unset)')
def push(model_path: str, existing_repo: str, revision: str, coldkey: str, hotkey: str, chutes_api_key: str):
    """Pushes a model to be hosted by your miner."""
    # -----------------------------------------------------------------------------
    # 1. Wallet & config
    # -----------------------------------------------------------------------------
    coldkey = coldkey or get_conf("BT_WALLET_COLD", "default")
    hotkey  = hotkey  or get_conf("BT_WALLET_HOT", "default")
    logger.debug("Using coldkey=%s, hotkey=%s", coldkey, hotkey)
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    # Required API credentials
    hf_user        = get_conf("HF_USER")
    hf_token       = get_conf("HF_TOKEN")
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    chute_user     = get_conf("CHUTE_USER")
    # TODO: validate API creds, exit gracefully if missing

    # -----------------------------------------------------------------------------
    # 2. Prepare HF repo name - If --existing-repo provided, use it directly and skip local upload
    # -----------------------------------------------------------------------------
    repo_name = existing_repo or f"{hf_user}/Affine-{wallet.hotkey.ss58_address}"
    logger.debug("Using existing HF repo: %s" if existing_repo else "Hugging Face repo: %s", repo_name)

    # -----------------------------------------------------------------------------
    # 3. Create & secure HF repo
    # -----------------------------------------------------------------------------
    api = HfApi(token=hf_token)
    if not existing_repo:
        api.create_repo(repo_id=repo_name, repo_type="model", private=True, exist_ok=True)
        try: api.update_repo_visibility(repo_id=repo_name, private=True)
        except Exception: logger.debug("Repo already private or visibility update failed")

    # -----------------------------------------------------------------------------
    # 4. Upload model files to HF (skip if using existing repo)
    # -----------------------------------------------------------------------------
    async def deploy_model_to_hf():
        logger.debug("Starting model upload from %s", model_path)
        # Gather files
        files = []
        for root, _, fnames in os.walk(model_path):
            if ".cache" in root or any(p.startswith(".") for p in root.split(os.sep)):
                continue
            for fname in fnames:
                if not (fname.startswith(".") or fname.endswith(".lock")):
                    files.append(os.path.join(root, fname))

        # Upload files with limited concurrency to avoid HF 429 errors
        SEM = asyncio.Semaphore(int(os.getenv("AFFINE_UPLOAD_CONCURRENCY", "2")))

        async def _upload(path: str):
            rel = os.path.relpath(path, model_path)
            async with SEM:  # limit concurrent commits
                await asyncio.to_thread(
                    lambda: api.upload_file(
                        path_or_fileobj=path,
                        path_in_repo=rel,
                        repo_id=repo_name,
                        repo_type="model"
                    )
                )
                logger.debug("Uploaded %s", rel)

        await asyncio.gather(*(_upload(p) for p in files))
        logger.debug("Model upload complete (%d files)", len(files))

    asyncio.run(deploy_model_to_hf()) if not existing_repo else logger.debug("Skipping model upload because --existing-repo was provided")

    # -----------------------------------------------------------------------------
    # 5. Fetch latest revision hash
    # -----------------------------------------------------------------------------
    if revision:
        logger.debug("Using user-supplied revision: %s", revision)
    else:
        info      = api.repo_info(repo_id=repo_name, repo_type="model")
        revision  = getattr(info, "sha", getattr(info, "oid", "")) or ""
        logger.debug("Latest revision from HF: %s", revision)

    # -----------------------------------------------------------------------------
    # 6. Commit model revision on-chain
    # -----------------------------------------------------------------------------
    chute_id = None

    async def commit_to_chain():
        """Submit the model commitment, retrying on quota errors."""
        logger.debug("Preparing on-chain commitment")
        sub     = await get_subtensor()
        payload = json.dumps({"model": repo_name, "revision": revision, "chute_id": chute_id})
        while True:
            try:
                await sub.set_reveal_commitment(wallet=wallet, netuid=NETUID, data=payload, blocks_until_reveal=1)
                logger.debug("On-chain commitment submitted")
                break
            except MetadataError as e:
                if "SpaceLimitExceeded" in str(e):
                    logger.debug("SpaceLimitExceeded – waiting one block before retrying")
                    await sub.wait_for_block()
                else:
                    raise


    # -----------------------------------------------------------------------------
    # 7. Make HF repo public
    # -----------------------------------------------------------------------------
    try:
        api.update_repo_visibility(repo_id=repo_name, private=False)
        logger.debug("Repo made public")
    except Exception:
        logger.trace("Failed to make repo public (already public?)")

    # -----------------------------------------------------------------------------
    # 8. Deploy Chute
    # -----------------------------------------------------------------------------
    async def deploy_to_chutes():
        logger.debug("Building Chute config")
        rev_flag = f'revision="{revision}",' if revision else ""
        chutes_config = textwrap.dedent(f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_name}",
    model_name="{repo_name}",
    image="chutes/sglang:0.4.9.post3",
    concurrency=20,
    {rev_flag}
    node_selector=NodeSelector(
        gpu_count=8,
        min_vram_gb_per_gpu=24,
    ),
    engine_args=(
        "--trust-remote-code "
    ),
)
""")
        tmp_file = Path("tmp_chute.py")
        tmp_file.write_text(chutes_config)
        logger.debug("Wrote Chute config to %s", tmp_file)
        logger.debug("=== chute file ===\n%s", tmp_file.read_text())

        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--public"]
        env = {**os.environ, "CHUTES_API_KEY": chutes_api_key}
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE,
        )
        # Auto-answer the interactive Y/N prompt
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()
        stdout, _ = await proc.communicate()
        output = stdout.decode().split('confirm? (y/n)')[1].strip()
        logger.trace(output)

        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\|\s+(\w+)', output)
        if match and match.group(2) == "ERROR":
            logger.debug("Chutes deploy failed with the above error log")
            raise RuntimeError("Chutes deploy failed")
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    # -----------------------------------------------------------------------------
    # 8b. Retrieve chute_id and commit on-chain
    # -----------------------------------------------------------------------------
    chute_id = asyncio.run(get_latest_chute_id(repo_name, api_key=chutes_api_key))

    asyncio.run(commit_to_chain())

    # -----------------------------------------------------------------------------
    # 9. Warm up model until it’s marked hot
    # -----------------------------------------------------------------------------
    async def warmup_model():
        logger.debug("Warming up model with SAT challenges")
        sub       = await get_subtensor()
        meta      = await sub.metagraph(NETUID)
        my_uid    = meta.hotkeys.index(wallet.hotkey.ss58_address)
        miner  = (await miners(netuid=NETUID))[my_uid]

        while not (miner.chute or {}).get("hot", False):
            challenge = await SAT().generate()
            await run(challenges=challenge, miners=[miner])
            await sub.wait_for_block()
            miner = (await miners(netuid=NETUID))[my_uid]
            logger.trace("Checked hot status: %s", (miner.chute or {}).get("hot"))

        logger.debug("Model is now hot and ready")

    asyncio.run(warmup_model())
    logger.debug("Mining setup complete. Model is live!")  
