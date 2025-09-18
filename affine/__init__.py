
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
_GATING_LOCKS: Dict[int, asyncio.Lock] = {}
GATING_TTL = 3600  # 60 min
WEIGHTS_SHA_CACHE: Dict[Tuple[str, Optional[str]], Tuple[Optional[set[str]], float]] = {}
_WEIGHTS_LOCKS: Dict[int, asyncio.Lock] = {}
WEIGHTS_TTL = 3600  # 60 min

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

def _get_weights_lock() -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    lock = _WEIGHTS_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _WEIGHTS_LOCKS[key] = lock
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
    weights_shas: Optional[set[str]] = None
    

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
# Global buffer to reduce R2 writes; flush on threshold or force.
SINK_BUFFER: list["Result"] = []
SINK_BUFFER_SIZE = int(os.getenv("AFFINE_SINK_BUFFER", "2000"))
async def sink_enqueue(wallet, block, results, force: bool = False):
    global SINK_BUFFER
    SINK_BUFFER.extend(results)
    if not force and len(SINK_BUFFER) < SINK_BUFFER_SIZE: return
    buf, SINK_BUFFER = SINK_BUFFER, []
    await sink(wallet=wallet, results=buf, block=block)
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

async def get_weights_shas(model_id: str, revision: Optional[str] = None) -> Optional[set[str]]:
    key = (model_id, revision)
    now = time.time()
    cached = WEIGHTS_SHA_CACHE.get(key)
    if cached and now - cached[1] < WEIGHTS_TTL:
        return cached[0]
    async with _get_weights_lock():
        cached = WEIGHTS_SHA_CACHE.get(key)
        if cached and now - cached[1] < WEIGHTS_TTL:
            return cached[0]
        try:
            def _repo_info():
                return HfApi(token=os.getenv("HF_TOKEN")).repo_info(
                    repo_id=model_id, repo_type="model", revision=revision, files_metadata=True
                )
            info = await asyncio.to_thread(_repo_info)
            sib = getattr(info, "siblings", None) or []
            def _name(s): return getattr(s, "rfilename", None) or getattr(s, "path", "")
            shas = {str(getattr(s, "lfs", {})["sha256"]) for s in sib
                    if (isinstance(getattr(s, "lfs", None), dict) and _name(s).endswith(".safetensors") and "sha256" in getattr(s, "lfs", {}))}
            WEIGHTS_SHA_CACHE[key] = (shas or None, now)
            return shas or None
        except Exception as e:
            logger.trace(f"HF weights sha lookup failed for {model_id}@{revision}: {e}")
            WEIGHTS_SHA_CACHE[key] = (None, now)
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
    meta_sem = asyncio.Semaphore(int(os.getenv("AFFINE_META_CONCURRENCY", "12")))
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
                async with meta_sem:
                    shas = await get_weights_shas(model, miner_revision)
                return Miner(
                    uid=uid, hotkey=hotkey, model=model, block=int(block),
                    revision = miner_revision,
                    slug = slug,
                    chute=chute,
                    weights_shas=shas,
                )
        except: pass
    results = await asyncio.gather(*(fetch(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}
    # Remove duplicates.
    if output:
        earliest_by_sha: Dict[str, Tuple[int, int]] = {}
        for uid, m in output.items():
            if not m.weights_shas: continue
            blk = m.block if isinstance(m.block, int) else (int(m.block) if m.block is not None else (2**63 - 1))
            for sha in m.weights_shas:
                prev = earliest_by_sha.get(sha)
                if prev is None or blk < prev[0]:
                    earliest_by_sha[sha] = (blk, uid)
        if earliest_by_sha:
            keep = set(output.keys())
            for uid, m in output.items():
                if m.weights_shas and any(earliest_by_sha.get(s, (None, uid))[1] != uid for s in m.weights_shas):
                    if uid in keep: keep.remove(uid)
            output = {uid: m for uid, m in output.items() if uid in keep}
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
                            await sink_enqueue(wallet, blk, flat)
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
                            await sink_enqueue(wallet, blk, flat, force=True)
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
    
# --- Scoring hyperparameters --------------------------------------------------
TAIL = 10_000
ALPHA = 0.9
EPS_FLOOR   = 0.005
Z_NOT_WORSE = 1.28
EPS_WIN     = 0.008
Z_WIN       = 0.5
ELIG        = 0.03 

async def get_weights(tail: int = TAIL, scale: float = 1):
    """
    Compute miner weights using ε-Pareto dominance and combinatoric subset winners.

    Pipeline
      1) Ingest last `tail` blocks → per-miner per-env accuracy.
      2) Determine eligibility (>=90% of per-env max count AND must be queryable).
      3) Global ε-dominance (all envs) for canonical 'best' (for tie breaks / summaries).
      4) Combinatoric scoring:
           - For every non-empty subset S of ENVS, pick the ε-Pareto winner on S.
           - Award K_|S| where K_1 = scale, K_s = C(N, s-1)*K_{s-1}.
         Fallback if no dominance edges on S: earliest version (earlier block wins).
      5) Normalize scores over eligibles to produce weights. Metrics + summary emitted.

    Returns:
      (uids, weights): list of eligible UIDs (best last) and their weights (sum to 1).
    """

    # --- fetch + prune --------------------------------------------------------
    st = await get_subtensor()
    blk = await st.get_current_block()
    logger.info(f"Pruning {tail} blocks from {blk - tail} to {blk}")
    await prune(tail=tail)

    meta = await st.metagraph(NETUID)
    BASE_HK = meta.hotkeys[0]
    N_envs = len(ENVS)
    
    # --- get currently queryable miners ---------------------------------------
    queryable_miners = await miners(meta=meta)
    queryable_hks = {m.hotkey for m in queryable_miners.values()}
    logger.info(f"Found {len(queryable_hks)} queryable miners (hot, valid chute, not gated)")

    # Tallies for all known hotkeys (so metrics update is safe even if some have no data)
    cnt   = {hk: defaultdict(int)   for hk in meta.hotkeys}  # per-env counts
    succ  = {hk: defaultdict(int)   for hk in meta.hotkeys}  # per-env correct (0/1 or [0,1])
    prev  = {}                                                # last sample per hk
    v_id  = {}                                                # (model, revision) per hk
    first_block = {}                                          # earliest block for current version

    # Pre-seed earliest commit block per miner from on-chain commitments
    try:
        commits = await st.get_all_revealed_commitments(NETUID)
        for uid, hk in enumerate(meta.hotkeys):
            if hk in commits:
                blk, _ = commits[hk][-1]
                first_block[hk] = 0 if uid == 0 else int(blk)
    except Exception:
        pass

    # --- ingest ---------------------------------------------------------------
    logger.info(f"Loading data from {blk - tail} to {blk}")
    async for c in dataset(tail=tail):
        NRESULTS.inc()
        hk, env = c.miner.hotkey, c.challenge.env.name

        # keep the base hk; otherwise require model family
        try:
            name = c.miner.model.split("/", 1)[1].lower()
        except Exception:
            name = str(c.miner.model).lower()
        if hk not in cnt or (hk != BASE_HK and not name.startswith("affine")):
            continue

        cur_vid = (c.miner.model, c.miner.revision)

        # On version change, reset ALL env streams and timestamp to current block
        if v_id.get(hk) != cur_vid:
            v_id[hk] = cur_vid
            first_block[hk] = c.miner.block
            for e in ENVS:
                cnt[hk][e] = 0
                succ[hk][e] = 0
        else:
            # Keep earliest commit block for the active version
            try:
                fb = int(first_block.get(hk, c.miner.block)) if first_block.get(hk) is not None else int(c.miner.block)
                cb = int(c.miner.block) if c.miner.block is not None else fb
                first_block[hk] = fb if fb <= cb else cb
            except Exception:
                pass

        # accumulate on successes.
        prev[hk] = c
        if c.response.success:
            cnt[hk][env]  += 1
            succ[hk][env] += float(c.evaluation.score)

    logger.info("Collected results.")

    if not prev:
        logger.warning("No results collected; defaulting to uid 0")
        return [0], [1.0]

    # --- accuracy + MAXENV ----------------------------------------------------
    acc = {
        hk: {e: (succ[hk][e] / cnt[hk][e] if cnt[hk][e] else 0.0) for e in ENVS}
        for hk in meta.hotkeys
    }

    active_hks = list(prev.keys())
    for e in ENVS:
        max_e = max((acc[hk][e] for hk in active_hks), default=0.0)
        MAXENV.labels(env=e).set(max_e)
    logger.info("Computed accuracy & updated MAXENV.")

    # --- eligibility: must be queryable AND meet sample requirements ---------
    required = {}
    for e in ENVS:
        max_cnt = max((cnt[hk][e] for hk in active_hks), default=0)
        max_cnt = min(max_cnt, 2000)
        required[e] = 10 + int(ELIG * max_cnt)
    # Only eligible if: 1) currently queryable, 2) meets sample requirements
    eligible = {hk for hk in active_hks 
                if hk in queryable_hks and all(cnt[hk][e] >= required[e] for e in ENVS)}
    logger.info(f"Eligible miners: {len(eligible)} (from {len(active_hks)} active, {len(queryable_hks)} queryable)")

    # --- ε-Pareto dominance helpers ------------------------------------------
    def thr_not_worse(a_i: float, n_i: int, a_j: float, n_j: int) -> float:
        """Tolerance for 'not worse' on an env: max(EPS_FLOOR, Z * SE_diff) with capped n to blunt volume advantage."""
        if Z_NOT_WORSE <= 0:
            return EPS_FLOOR
        n_i_eff = min(int(n_i), 1000)
        n_j_eff = min(int(n_j), 1000)
        var = (a_i * (1 - a_i)) / max(n_i_eff, 1) + (a_j * (1 - a_j)) / max(n_j_eff, 1)
        return max(EPS_FLOOR, Z_NOT_WORSE * math.sqrt(var))

    def thr_better(a_i: float, n_i: int, a_j: float, n_j: int, nw: float) -> float:
        """
        Margin to claim 'better on at least one env'. Kept ≤ 'not worse' tolerance.
        Floor-based by default; set Z_WIN>0 to scale with SE_diff.
        """
        if Z_WIN > 0:
            n_i_eff = min(int(n_i), 1000)
            n_j_eff = min(int(n_j), 1000)
            var = (a_i * (1 - a_i)) / max(n_i_eff, 1) + (a_j * (1 - a_j)) / max(n_j_eff, 1)
            t = max(EPS_WIN, Z_WIN * math.sqrt(var))
        else:
            t = EPS_WIN
        return min(t, nw)

    def dominates_on(a: str, b: str, subset) -> bool:
        """
        True iff 'a' is not-worse than 'b' on every env in `subset` (within thr_not_worse),
        and strictly better on at least one env by thr_better. Full ε-ties break by earlier start.
        """
        not_worse_all = True
        better_any    = False
        tie_all       = True
        for e in subset:
            ai, aj = acc[a][e], acc[b][e]
            ni, nj = cnt[a][e], cnt[b][e]
            nw  = thr_not_worse(ai, ni, aj, nj)
            bet = thr_better(ai, ni, aj, nj, nw)

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

    # Global dominance (full ENVS) for summary + canonical "best"
    dom_full = defaultdict(int)
    # Pool for dominance should be eligible miners, or queryable miners if no eligible ones
    pool_for_dom = eligible if eligible else (queryable_hks & set(active_hks))
    for a, b in itertools.permutations(pool_for_dom, 2):
        if dominates_on(a, b, ENVS):
            dom_full[a] += 1
    logger.info("Computed ε-dominance counts (full env set).")

    def ts(hk: str) -> int:
        """Block-number timestamp; prefer earliest commit"""
        return int(first_block[hk]) if hk in first_block else float('inf')

    # Best should be from queryable miners only
    best_candidates = pool_for_dom if pool_for_dom else (queryable_hks if queryable_hks else active_hks[:1])
    best = max(best_candidates, key=lambda hk: (dom_full.get(hk, 0), -ts(hk))) if best_candidates else active_hks[0]
    best_uid = meta.hotkeys.index(best)

    # --- combinatoric scoring over all non-empty env subsets ------------------
    def layer_weights(N: int, kappa: float):
        """Per-subset weights K_s: K_1=kappa; K_s=C(N,s-1)*K_{s-1} for s>=2."""
        K = {1: kappa}
        for s in range(2, N + 1):
            K[s] = kappa * (2**s)
        return K

    def subset_winner(env_subset):
        """
        Winner on env_subset via ε-Pareto. If no dominance edges, fall back to:
          earliest version start block (earlier block wins).
        """
        dom_local = defaultdict(int)
        for x, y in itertools.permutations(pool_for_dom, 2):
            if dominates_on(x, y, env_subset):
                dom_local[x] += 1

        return max(pool_for_dom, key=lambda hk: (dom_local.get(hk, 0), -ts(hk)))

    # Calculate combinatoric scores for all miners (not just eligible)
    K = layer_weights(N_envs, scale)
    score = defaultdict(float)
    layer_points = {hk: defaultdict(float) for hk in active_hks}

    # --- Find single-env winners for highlighting ----------------------------
    env_winners = {}
    for e in ENVS:
        env_winners[e] = subset_winner((e,))

    # Award K_s to each subset winner
    for s in range(1, N_envs + 1):
        for env_subset in itertools.combinations(ENVS, s):
            w = subset_winner(env_subset)
            score[w] += K[s]
            layer_points[w][s] += K[s]

    # If no eligible miners exist, fall back to uid 0 with weight 1.0.
    if not eligible:
        logger.warning(f"No eligible miners (queryable={len(queryable_hks)}); assigning weight 1.0 to uid 0.")
        for uid, hk in enumerate(meta.hotkeys):
            WEIGHT.labels(uid=uid).set(1.0 if uid == 0 else 0.0)
            for e in ENVS:
                a = acc[hk][e]
                if a > 0:
                    SCORE.labels(uid=uid, env=e).set(a)

        hdr = (
            ["UID", "Model", "Rev"]
            + [f"{e}" for e in ENVS]
            + [f"L{s}" for s in range(1, N_envs + 1)]
            + ["Pts", "Elig", "Wgt"]
        )
        def row(hk: str):
            m = prev[hk].miner
            w = 1.0 if hk == best else 0.0
            model_name = str(m.model)[:50]
            env_cols = []
            for e in ENVS:
                base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}"
                if hk == env_winners.get(e):
                    env_cols.append(f"*{base}*")
                else:
                    env_cols.append(base)
            return [
                m.uid, model_name, str(m.revision)[:5],
                *env_cols,
                *[f"{layer_points[hk][s]:.1f}" for s in range(1, N_envs + 1)],
                f"{score.get(hk, 0.0):.2f}",
                "Y" if hk in eligible else "N",
                f"{w:.4f}",
            ]
        rows = sorted((row(hk) for hk in active_hks), key=lambda r: (r[-3], r[0]), reverse=True)
        print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))
        return [0], [1.0]  # Always return uid 0 when no eligible miners

    # Eligible path: normalize scores to weights over the eligible pool only
    total_points = sum(score[hk] for hk in eligible)
    if total_points <= 0:
        logger.warning("Combinatoric scoring returned zero total; falling back to canonical best.")
        weight_by_hk = {hk: (1.0 if hk == best else 0.0) for hk in eligible}
    else:
        weight_by_hk = {hk: (score[hk] / total_points) for hk in eligible}

    # --- summary printout -----------------------------------------------------
    hdr = (
        ["UID", "Model", "Rev"]
        + [f"{e}" for e in ENVS]
        + [f"L{s}" for s in range(1, N_envs + 1)]
        + ["Pts", "Elig", "Wgt"]
    )
    def row(hk: str):
        m = prev[hk].miner
        w = weight_by_hk.get(hk, 0.0)
        model_name = str(m.model)[:50]
        env_cols = []
        for e in ENVS:
            base = f"{100 * acc[hk][e]:.2f}/{cnt[hk][e]}"
            if hk == env_winners.get(e):
                env_cols.append(f"*{base}*")
            else:
                env_cols.append(base)
        return [
            m.uid, model_name[:30], str(m.revision)[:5],
            *env_cols,
            *[f"{layer_points[hk][s]:.1f}" for s in range(1, N_envs + 1)],
            f"{score.get(hk, 0.0):.2f}",
            "Y" if hk in eligible else "N",
            f"{w:.4f}",
        ]
    ranked_rows   = sorted((row(hk) for hk in eligible), key=lambda r: float(r[-3]), reverse=True)
    unranked_rows = sorted((row(hk) for hk in active_hks if hk not in eligible), key=lambda r: float(r[-3]), reverse=True)
    rows = ranked_rows + unranked_rows
    print("Validator Summary:\n" + tabulate(rows, hdr, tablefmt="plain"))

    # --- Prometheus updates ---------------------------------------------------
    for uid, hk in enumerate(meta.hotkeys):
        WEIGHT.labels(uid=uid).set(weight_by_hk.get(hk, 0.0))
        for e in ENVS:
            a = acc[hk][e]
            if a > 0:
                SCORE.labels(uid=uid, env=e).set(a)

    # --- Return weights in a stable shape (best last, as before) -------------
    eligible_uids = [meta.hotkeys.index(hk) for hk in eligible]
    uids = [u for u in eligible_uids if u != best_uid] + [best_uid]
    weights = [weight_by_hk.get(meta.hotkeys[u], 0.0) for u in uids]
    return uids, weights


        
@cli.command("validate")
def validate():
    global HEARTBEAT
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
                HEARTBEAT = time.monotonic()
                if subtensor is None: subtensor = await get_subtensor()
                BLOCK = await subtensor.get_current_block()
                if BLOCK % TEMPO != 0 or BLOCK <= LAST: 
                    logger.debug(f'Waiting ... {BLOCK} % {TEMPO} == {BLOCK % TEMPO} != 0')
                    await subtensor.wait_for_block()
                    continue
                
                # ---------------- Set weights. ------------------------
                uids, weights = await get_weights()
        
                # ---------------- Set weights. ------------------------
                logger.info("Setting weights ...")
                await retry_set_weights( wallet, uids=uids, weights=weights, retry = 3)
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
            watchdog(timeout = (60 * 20))
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
