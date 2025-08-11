
#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import os
import re
import sys
import json
import time
import click
import random
import hashlib
import aiohttp
import asyncio
import logging
import textwrap
import traceback
import itertools
from .utils import *
import datetime as dt
from datetime import datetime
from tqdm import tqdm
import bittensor as bt
import datasets as hf_ds                    
from pathlib import Path
from tqdm.asyncio import tqdm
from tabulate import tabulate
from dotenv import load_dotenv
from typing import AsyncIterator
from huggingface_hub import HfApi
from botocore.config import Config
from collections import defaultdict
from abc import ABC, abstractmethod
from aiobotocore.session import get_session
from huggingface_hub import snapshot_download
from bittensor.core.errors import MetadataError
from pydantic import BaseModel, Field, validator, ValidationError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable
from pydantic import root_validator
from aiohttp import ClientConnectorError
import socket
from urllib.parse import urlparse
import requests

__version__ = "0.0.0"

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
NETUID = 120
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")

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
# Model SHA cache for weight deduplication
MODEL_SHA_CACHE = {}  # {model_id: (sha, last_checked)}
# Model first uploader tracking {model_sha: (hotkey, block_uploaded)}
MODEL_FIRST_UPLOADER = {}
# Replace global loop-bound lock with per-event-loop lazy locks to avoid cross-loop errors
_GATING_LOCKS: Dict[int, asyncio.Lock] = {}
_SHA_LOCKS: Dict[int, asyncio.Lock] = {}
GATING_TTL = 300  # 5 minutes
SHA_TTL = 3600  # 1 hour

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

def _get_sha_lock() -> asyncio.Lock:
    """Return an asyncio.Lock bound to the current running loop for SHA operations."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Fallback if called when no loop is running yet
        loop = asyncio.get_event_loop()
    key = id(loop)
    lock = _SHA_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _SHA_LOCKS[key] = lock
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

async def check_model_gated(model_id: str) -> Optional[bool]:
    """Check if model is gated, with caching."""
    async with _get_gating_lock():
        now = time.time()
        if model_id in MODEL_GATING_CACHE:
            is_gated, ts = MODEL_GATING_CACHE[model_id]
            if now - ts < GATING_TTL:
                return is_gated
        
        # Check HF API
        try:
            r = await asyncio.to_thread(
                requests.get, f"https://huggingface.co/api/models/{model_id}", timeout=5
            )
            if r.status_code == 200:
                is_gated = r.json().get("gated", False)
                MODEL_GATING_CACHE[model_id] = (is_gated, now)
                return is_gated
        except Exception as e:
            logger.trace(f"Gate check failed for {model_id}: {e}")
        
        # Use cached value if available
        if model_id in MODEL_GATING_CACHE:
            is_gated, _ = MODEL_GATING_CACHE[model_id]
            MODEL_GATING_CACHE[model_id] = (is_gated, now)  # Update timestamp
            return is_gated
        return None

async def get_model_weights_sha(model_id: str, revision: Optional[str] = None) -> Optional[str]:
    """Get SHA of actual model weight files from Hugging Face, with caching."""
    cache_key = f"{model_id}@{revision or 'main'}"
    async with _get_sha_lock():
        now = time.time()
        if cache_key in MODEL_SHA_CACHE:
            sha, ts = MODEL_SHA_CACHE[cache_key]
            if now - ts < SHA_TTL:
                return sha
        
        # Get actual model file hashes instead of git commit SHA
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # First get list of all files in the repo
            files = await asyncio.to_thread(
                api.list_repo_files,
                repo_id=model_id,
                repo_type="model", 
                revision=revision
            )
            
            # Filter to main model weight files (safetensors, bin files)
            model_files = []
            for f in files:
                # Include main model weight files, exclude other files like config.json, tokenizer files, etc.
                if (f.endswith('.safetensors') or f.endswith('.bin')) and \
                   ('model' in f.lower() or 'pytorch_model' in f.lower()):
                    # Exclude specific non-weight files
                    if not any(exclude in f.lower() for exclude in ['tokenizer', 'config', 'generation', 'training']):
                        model_files.append(f)
            
            if not model_files:
                logger.trace(f"No model weight files found for {model_id}@{revision}")
                return None
                
            # Get file info including SHA256 hashes
            paths_info = await asyncio.to_thread(
                api.get_paths_info,
                repo_id=model_id,
                paths=model_files,
                repo_type="model",
                revision=revision
            )
            
            # Extract and combine SHA256 hashes of actual model files
            file_hashes = []
            for info in paths_info:
                if hasattr(info, 'lfs') and info.lfs and hasattr(info.lfs, 'sha256'):
                    file_hashes.append(f"{info.path}:{info.lfs.sha256}")
                    logger.trace(f"Model file {info.path}: {info.lfs.sha256}")
                else:
                    logger.trace(f"No LFS SHA256 found for {info.path}")
            
            if not file_hashes:
                logger.trace(f"No file hashes available for {model_id}@{revision}")
                return None
            
            # Create combined hash of all model file hashes
            # Sort to ensure consistent ordering
            combined_hash_input = "|".join(sorted(file_hashes))
            weights_sha = hashlib.sha256(combined_hash_input.encode()).hexdigest()
            
            MODEL_SHA_CACHE[cache_key] = (weights_sha, now)
            logger.trace(f"Got model weights SHA for {model_id}@{revision}: {weights_sha} (from {len(file_hashes)} files)")
            return weights_sha
            
        except Exception as e:
            logger.trace(f"Model weights SHA check failed for {model_id}@{revision}: {e}")
        
        # Use cached value if available
        if cache_key in MODEL_SHA_CACHE:
            sha, _ = MODEL_SHA_CACHE[cache_key]
            MODEL_SHA_CACHE[cache_key] = (sha, now)  # Update timestamp
            return sha
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
        await SUBTENSOR.initialize()
        logger.trace("Connected")
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
        from .envs.sat import SAT
        from .envs.abd import ABD
        from .envs.ded import DED
        ENVS = {"SAT": SAT, "ABD": ABD, "DED": DED}
        return ENVS[v]() if isinstance(v, str) else v
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
        from .envs.sat import SAT
        from .envs.abd import ABD
        from .envs.ded import DED
        ENVS = {"SAT": SAT, "ABD": ABD, "DED": DED}
        return ENVS[v]() if isinstance(v, str) else v
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

# Real import.    
from .envs.sat import SAT
from .envs.abd import ABD
from .envs.ded import DED
ENVS = {"SAT": SAT, "ABD": ABD, "DED": DED}

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
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

get_client_ctx = lambda: get_session().create_client(
    "s3", endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
    config=Config(max_pool_connections=256)
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
        r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
        return json.loads(await r["Body"].read())

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

# ── Minimal sink / misc helpers (optional) ──────────────────────────────────
async def sink(wallet: bt.wallet, results: list["Result"], block: int = None):
    if not results: return
    if block is None:
        sub = await get_subtensor(); block = await sub.get_current_block()
    key = f"{RESULT_PREFIX}{_w(block):09d}-{wallet.hotkey.ss58_address}.json"
    new = [r.sign(wallet) or r.model_dump(mode="json") for r in results]
    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=key)
            merged = json.loads(await r["Body"].read()) + new
        except c.exceptions.NoSuchKey:
            merged = new
        await c.put_object(Bucket=FOLDER, Key=key, Body=_dumps(merged),
                           ContentType="application/json")
    if len(merged) == len(new):              # shard was new
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
        sem = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
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
        client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
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
        # Check gating status before querying
        if miner.model:
            is_gated = await check_model_gated(miner.model)
            if is_gated is None or is_gated:
                err = "Unknown model gating status" if is_gated is None else "Model is gated"
                logger.trace(f"Miner {miner.uid} - {err} for model {miner.model}")
                resp = Response(response=None, latency_seconds=0, attempts=0, model=miner.model, error=err, success=False)
                ev = Evaluation(env=chal.env, score=0.0, extra={"error": err, "gated": is_gated})
                return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)
        
        # Normal processing for non-gated models
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
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
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
    async def fetch(uid: int):
        try:
            hotkey = meta.hotkeys[ uid ]
            if hotkey not in commits: return None
            commit = commits[hotkey]
            block, data = commit[-1]     
            block = 0 if uid == 0 else block
            data = json.loads(data)
            model, miner_revision, chute_id = data.get("model"), data.get("revision"), data.get("chute_id")
            chute = await get_chute(chute_id)
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
    return output


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)
    
# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)
            
# --------------------------------------------------------------------------- #
#                               Runner                                        #
# --------------------------------------------------------------------------- #
@cli.command("runner")
def runner():    
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    async def _run():
        subtensor = None
        envs = { name: cls() for name, cls in ENVS.items() }
        while True:
            global HEARTBEAT
            try:
                if subtensor is None: subtensor = await get_subtensor()
                meta = await subtensor.metagraph( NETUID )
                blk = await subtensor.get_current_block()
                HEARTBEAT = time.monotonic()
                miners_map = await miners(meta=meta)
                challenges = [await e.generate() for e in envs.values()]
                results    = await run(challenges, miners_map, timeout=180)
                await sink( wallet = wallet, block = blk, results = results )
            except asyncio.CancelledError: break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in runner loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 10))
        )
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

        async def set_weights_handler(request: "web.Request"):
            try:
                logger.info("[signer] /set_weights: request received")
                payload = await request.json()
                netuid = int(payload.get('netuid', NETUID))
                uids = payload.get('uids') or []
                weights = payload.get('weights') or []
                wait_for_inclusion = bool(payload.get('wait_for_inclusion', False))
                coldkey = get_conf("BT_WALLET_COLD", "default")
                hotkey = get_conf("BT_WALLET_HOT", "default")
                wallet = bt.wallet(name=coldkey, hotkey=hotkey)
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

async def retry_set_weights( wallet: bt.Wallet, best_uid:int, retry: int = 10 ):
    # Delegate to signer; fallback to shared helper only if signer is unreachable
    signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
    try:
        logger.info(f"Calling signer at {signer_url} for set_weights uid={best_uid}")
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
                    "weights": [1.0],
                    "uids": [best_uid],
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
            wallet, NETUID, [best_uid], [1.0], False,
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
    
TAIL= 10_000
ALPHA = 0.9
async def get_weights(tail=TAIL):
    st = await get_subtensor()
    blk = await st.get_current_block()
    logger.info(f'Pruning {tail} blocks from {blk-tail} to {blk}')
    await prune(tail=tail)

    meta = await st.metagraph(NETUID)
    cnt  = {hk: defaultdict(int) for hk in meta.hotkeys}
    succ = {hk: defaultdict(int) for hk in meta.hotkeys}
    prev = {}

    logger.info(f'Loading data from {blk-tail} to {blk}')
    async for c in dataset(tail=tail):
        NRESULTS.inc()
        hk, env = c.miner.hotkey, c.challenge.env.name
        name = c.miner.model.split('/',1)[1].lower()
        BASE_HK = meta.hotkeys[0]
        if hk not in cnt or (hk != BASE_HK and not name.startswith('affine')):
            continue

        # reset if block/model/revision changed
        if hk in prev:
            p = prev[hk].miner
            if (p.block!=c.miner.block or p.model!=c.miner.model
             or p.revision!=c.miner.revision):
                succ[hk][env] = 0

        prev[hk] = c
        cnt[hk][env] += 1
        succ[hk][env] += c.evaluation.score

    logger.info("Collected results.")

    # Check for duplicate model weights and filter to earliest uploaders only
    logger.info("Checking for duplicate model weights...")
    sha_to_first_uploader = {}  # {sha: (hotkey, created_at_timestamp)}
    unique_miners = {}  # hotkeys that have unique models or were first to upload
    
    for hk in prev:
        miner = prev[hk].miner
        if not miner.model:
            continue
            
        try:
            # Get the SHA of the model weights
            model_sha = await get_model_weights_sha(miner.model, miner.revision)
            if not model_sha:
                logger.trace(f"Could not get SHA for {miner.model}@{miner.revision}, skipping duplicate check")
                unique_miners[hk] = miner
                continue
                
            # Get the chute creation timestamp for accurate ordering
            chute_created_at = None
            if miner.chute and miner.chute.get('created_at'):
                try:
                    chute_created_at = datetime.fromisoformat(miner.chute['created_at'].replace('Z', '+00:00'))
                except Exception as e:
                    logger.trace(f"Could not parse chute created_at for {hk}: {e}")
            
            # Fallback to block number if chute timestamp unavailable
            fallback_time = miner.block if chute_created_at is None else None
            comparison_time = chute_created_at or fallback_time
                
            # Check if this SHA has been seen before
            if model_sha in sha_to_first_uploader:
                existing_hk, existing_time = sha_to_first_uploader[model_sha]
                
                # Special override: Alphatao users are always considered first (super user)
                current_is_alphatao = miner.model and miner.model.startswith('Alphatao')
                existing_miner = prev.get(existing_hk)
                existing_is_alphatao = existing_miner and existing_miner.miner.model and existing_miner.miner.model.startswith('Alphatao')
                
                is_earlier = False
                if current_is_alphatao and not existing_is_alphatao:
                    # Current is Alphatao, existing is not - Alphatao wins
                    is_earlier = True
                elif not current_is_alphatao and existing_is_alphatao:
                    # Current is not Alphatao, existing is - existing Alphatao wins
                    is_earlier = False
                elif current_is_alphatao and existing_is_alphatao:
                    # Both are Alphatao - use normal timestamp comparison between them
                    if chute_created_at is not None and isinstance(existing_time, datetime):
                        is_earlier = chute_created_at < existing_time
                    elif fallback_time is not None and isinstance(existing_time, int):
                        is_earlier = fallback_time < existing_time
                    elif chute_created_at is not None and isinstance(existing_time, int):
                        is_earlier = True
                    elif fallback_time is not None and isinstance(existing_time, datetime):
                        is_earlier = False
                else:
                    # Neither is Alphatao - use normal timestamp comparison
                    if chute_created_at is not None and isinstance(existing_time, datetime):
                        is_earlier = chute_created_at < existing_time
                    elif fallback_time is not None and isinstance(existing_time, int):
                        is_earlier = fallback_time < existing_time
                    elif chute_created_at is not None and isinstance(existing_time, int):
                        # Current has timestamp, existing has block - prefer timestamp (assume earlier)
                        is_earlier = True
                    elif fallback_time is not None and isinstance(existing_time, datetime):
                        # Current has block, existing has timestamp - prefer timestamp (assume later)
                        is_earlier = False
                    
                if is_earlier:
                    # This miner uploaded earlier, replace the previous one
                    time_desc = chute_created_at.isoformat() if chute_created_at else f"block {fallback_time}"
                    existing_desc = existing_time.isoformat() if isinstance(existing_time, datetime) else f"block {existing_time}"
                    
                    # Add special logging for Alphatao override
                    if current_is_alphatao and not existing_is_alphatao:
                        logger.info(f"Alphatao super user override for SHA {model_sha[:8]}: {hk} (Alphatao) replaces {existing_hk} (non-Alphatao)")
                    else:
                        logger.info(f"Found earlier uploader for SHA {model_sha[:8]}: {hk} ({time_desc}) vs {existing_hk} ({existing_desc})")
                    
                    if existing_hk in unique_miners:
                        del unique_miners[existing_hk]
                        # Reset scores for the replaced miner
                        for env in ENVS:
                            succ[existing_hk][env] = 0
                            cnt[existing_hk][env] = 0
                    sha_to_first_uploader[model_sha] = (hk, comparison_time)
                    unique_miners[hk] = miner
                else:
                    # This miner uploaded later, don't count their scores
                    time_desc = chute_created_at.isoformat() if chute_created_at else f"block {fallback_time}"
                    existing_desc = existing_time.isoformat() if isinstance(existing_time, datetime) else f"block {existing_time}"
                    
                    # Add special logging for when non-Alphatao is blocked by Alphatao
                    if not current_is_alphatao and existing_is_alphatao:
                        logger.info(f"Found duplicate model weights SHA {model_sha[:8]}: {hk} (non-Alphatao) blocked by existing Alphatao super user {existing_hk}, excluding from rewards")
                    else:
                        logger.info(f"Found duplicate model weights SHA {model_sha[:8]}: {hk} ({time_desc}) uploaded after {existing_hk} ({existing_desc}), excluding from rewards")
                    
                    # Reset scores for the duplicate miner
                    for env in ENVS:
                        succ[hk][env] = 0
                        cnt[hk][env] = 0
            else:
                # First time seeing this SHA
                sha_to_first_uploader[model_sha] = (hk, comparison_time)
                unique_miners[hk] = miner
                
        except Exception as e:
            logger.trace(f"Error checking SHA for {miner.model}@{miner.revision}: {e}")
            # On error, allow the miner to participate (fail open)
            unique_miners[hk] = miner
    
    logger.info(f"After duplicate filtering: {len(unique_miners)} unique miners from {len(prev)} total")

    # compute accuracy & maxenv
    acc = {
        hk: {e: (float(succ[hk][e])/cnt[hk][e] if cnt[hk][e] else 0)
             for e in ENVS}
        for hk in meta.hotkeys
    }
    max_acc = {}
    for e in ENVS:
        max_acc[e] = max(acc[hk][e] for hk in meta.hotkeys)
        MAXENV.labels(env=e).set(max_acc[e])
    logger.info("Computed accuracy & updated MAXENV.")

    # compute ranks with dense tie handling
    ranks = {}
    for e in ENVS:
        uniq = sorted({acc[h][e] for h in meta.hotkeys}, reverse=True)
        rank_of = {v: i+1 for i, v in enumerate(uniq)}
        ranks[e] = {h: rank_of[acc[h][e]] for h in meta.hotkeys}
    logger.info("Computed ranks.")

    # pairwise dominance
    dom = defaultdict(int)
    for a, b in itertools.permutations(meta.hotkeys, 2):
        if all(ranks[e][a] <= ranks[e][b] for e in ENVS) \
        and any(ranks[e][a] < ranks[e][b] for e in ENVS):
            dom[a] += 1
    logger.info("Computed dominance counts.")

    # select best non-gated model from unique miners only, fallback to any unique if none available
    non_gated_candidates = []
    for hk in unique_miners:
        if hk not in prev:  # Skip if not in prev (shouldn't happen)
            continue
        miner = prev[hk].miner
        if miner.model:
            is_gated = await check_model_gated(miner.model)
            if is_gated is False:  # Only add if explicitly non-gated
                non_gated_candidates.append(hk)
    
    # Select best from non-gated unique models, or fallback to all unique models
    if non_gated_candidates:
        best = max(non_gated_candidates, key=lambda hk: (dom[hk], -prev[hk].miner.block))
        logger.info(f"Selected non-gated unique model for weights")
    elif unique_miners:
        logger.warning("No non-gated unique models found, selecting from all unique available models")
        best = max(unique_miners.keys(), key=lambda hk: (dom[hk], -prev[hk].miner.block))
    else:
        logger.warning("No unique models found, falling back to all available models")
        best = max(prev, key=lambda hk: (dom[hk], -prev[hk].miner.block))
    
    best_uid = meta.hotkeys.index(best)

    # print summary
    hdr = ["UID","Model","Rev"] \
        + [f"{e}Acc" for e in ENVS] \
        + [f"{e}Rnk" for e in ENVS] \
        + [f"{e}N"   for e in ENVS] \
        + ["Dom","Wgt"]
    rows = sorted([
        [m.uid, m.model, m.revision[:5]]
        + [f"{100*acc[hk][e]:.2f}" for e in ENVS]
        + [ranks[e][hk]     for e in ENVS]
        + [cnt[hk][e]       for e in ENVS]
        + [dom[hk], 1 if hk==best else 0]
        for hk, m in ((hk, prev[hk].miner) for hk in prev)
    ], key=lambda r: r[-2], reverse=True)
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    # update Prometheus
    for uid, hk in enumerate(meta.hotkeys):
        WEIGHT.labels(uid=uid).set(1 if hk==best else 0)
        for e in ENVS:
            a = acc[hk][e]
            if a > 0:
                SCORE.labels(uid=uid, env=e).set(a)
                RANK.labels(uid=uid, env=e).set(ranks[e][hk])

    return best_uid, best


        
@cli.command("validate")
def validate():
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)    
    async def _run():     
        LAST = 0
        TEMPO = 100
        subtensor = None
        while True:
            try:
                # ---------------- Wait for set weights. -----------------
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
                if subtensor is None: subtensor = await get_subtensor()
                BLOCK = await subtensor.get_current_block()
                if BLOCK % TEMPO != 0 or BLOCK <= LAST: 
                    logger.debug(f'Waiting ... {BLOCK} % {TEMPO} == {BLOCK % TEMPO} != 0')
                    await subtensor.wait_for_block()
                    continue
                
                # ---------------- Set weights. ------------------------
                winner_uid, _ = await get_weights()
        
                # ---------------- Set weights. ------------------------
                logger.info("Setting weights ...")
                await retry_set_weights( wallet, winner_uid, retry = 3)
                subtensor = await get_subtensor()
                SETBLOCK = await subtensor.get_current_block()
                LASTSET.set_function(lambda: SETBLOCK - LAST)
                LAST = BLOCK           
            
                # ---------------- Other telemetry ------------------------
                CACHE.set(sum( f.stat().st_size for f in CACHE_DIR.glob("*.jsonl") if f.is_file()))
                
            except asyncio.CancelledError: break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
            
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 10))
        )
    asyncio.run(main())
    
    
@cli.command("weights")
def weights():
    asyncio.run(get_weights())

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
