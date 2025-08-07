
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
import aiofiles
import textwrap
import traceback
import itertools
from .utils import *
import datetime as dt
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
CONCUR        = 25
WINDOW        = int(os.getenv("AFFINE_WINDOW", 20))
RESULT_PREFIX = "affine/results/"
INDEX_KEY     = "affine/index.json"
FOLDER        = os.getenv("R2_FOLDER",  "affine")
BUCKET        = os.getenv("R2_BUCKET_ID", "80f15715bb0b882c9e967c13e677ed7d")
ACCESS        = os.getenv("R2_WRITE_ACCESS_KEY_ID", "ff3f4f078019b064bfb6347c270bee4d")
SECRET        = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "a94b20516013519b2959cbbb441b9d1ec8511dce3c248223d947be8e85ec754d")
ENDPOINT      = f"https://{BUCKET}.r2.cloudflarestorage.com"
CACHE_DIR     = Path(os.getenv("AFFINE_CACHE_DIR", Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
def _w(b: int) -> int: return (b // WINDOW) * WINDOW

get_client_ctx = lambda: get_session().create_client(
    "s3", endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
    config=Config(max_pool_connections=256)
)

# ── JSON helpers ─────────────────────────────────────────────────────────────
try:
    import orjson as _json
    _loads, _dumps = _json.loads, _json.dumps
except ModuleNotFoundError:
    _loads = lambda b: json.loads(b.decode())
    _dumps = lambda o: json.dumps(o, separators=(",", ":")).encode()

# ── S3 index helpers ────────────────────────────────────────────────────────
async def _index() -> list[str]:
    async with get_client_ctx() as c:
        r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
        return json.loads(await r["Body"].read())
    
async def _update_index(key: str) -> None:
    async with get_client_ctx() as c:
        try:
            r   = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
            idx = set(json.loads(await r["Body"].read()))
        except c.exceptions.NoSuchKey: idx = set()
        if key not in idx:
            idx.add(key)
            await c.put_object(Bucket=FOLDER, Key=INDEX_KEY,
                               Body=_dumps(sorted(idx)),
                               ContentType="application/json")

# ── shard discovery ─────────────────────────────────────────────────────────
async def _shard_keys(tail: int) -> list[str]:
    sub  = await get_subtensor()
    cur  = await sub.get_current_block()
    need = {w for w in range(_w(cur - tail), _w(cur) + WINDOW, WINDOW)}
    keys = [k for k in await _index()
            if (h := Path(k).name.split("-", 1)[0]).isdigit()
            and int(h) in need]
    keys.sort()
    return keys

# ── cache helpers ───────────────────────────────────────────────────────────
async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    """Download a shard and store it as <CACHE_DIR>/<name>.jsonl."""
    out   = CACHE_DIR / f"{Path(key).name}.jsonl"
    stamp = out.with_suffix(".modified")
    async with sem, get_client_ctx() as c:
        if out.exists() and stamp.exists():
            if (await c.head_object(Bucket=FOLDER,
                                    Key=key))["LastModified"].isoformat() \
               == stamp.read_text().strip():
                return out
        obj  = await c.get_object(Bucket=FOLDER, Key=key)
        body = await obj["Body"].read()
        lm   = obj["LastModified"].isoformat()
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(b"\n".join(_dumps(i) for i in _loads(body)) + b"\n")
    os.replace(tmp, out); stamp.write_text(lm)
    return out

async def _jsonl(path: Path):
    """Async-iterate over lines in a local .jsonl file."""
    try:
        async with aiofiles.open(path, "rb") as f:
            async for l in f: yield l.rstrip(b"\n")
    except ModuleNotFoundError:
        for l in path.read_bytes().splitlines(): yield l

# ── public helpers ──────────────────────────────────────────────────────────
async def prefetch(tail: int, max_concurrency: int = CONCUR) -> None:
    """Download all shards for the trailing `tail` blocks into CACHE_DIR."""
    sem  = asyncio.Semaphore(max_concurrency)
    keys = await _shard_keys(tail)
    await asyncio.gather(*(_cache_shard(k, sem) for k in keys))

async def dataset(tail: int, max_concurrency: int = CONCUR) -> AsyncIterator["Result"]:
    """Yield verified Result objects in deterministic order."""
    keys  = await _shard_keys(tail)
    sem   = asyncio.Semaphore(max_concurrency)
    tasks = [asyncio.create_task(_cache_shard(k, sem)) for k in keys[:max_concurrency]]
    nxt   = max_concurrency
    bar   = tqdm(total=0, unit="res", dynamic_ncols=True, desc="Results")
    for i, key in enumerate(keys):
        path = await tasks[i]
        if nxt < len(keys):
            tasks.append(asyncio.create_task(_cache_shard(keys[nxt], sem)))
            nxt += 1
        async for raw in _jsonl(path):
            try:
                r = Result.model_validate(_loads(raw))
                if r.verify(): bar.update(1); yield r
            except Exception:
                pass
    bar.close()
    
async def sink(wallet: "bt.wallet", results: list["Result"], block: int | None = None) -> None:
    """Upload signed Result shard for `block` and update global index."""
    if not results: return
    sub = await get_subtensor()
    block = block or await sub.get_current_block()
    key   = f"{RESULT_PREFIX}{_w(block):09d}-{wallet.hotkey.ss58_address}.json"
    payload = [r.sign(wallet) or r.model_dump(mode="json") for r in results]
    async with get_client_ctx() as c:
        try:
            r        = await c.get_object(Bucket=FOLDER, Key=key)
            merged   = json.loads(await r["Body"].read()) + payload
        except c.exceptions.NoSuchKey:
            merged = payload
        await c.put_object(Bucket=FOLDER, Key=key,
                           Body=_dumps(merged),
                           ContentType="application/json")
    if len(merged) == len(payload):                       # new shard
        await _update_index(key)

# ── pruning helper (optional) ───────────────────────────────────────────────
async def prune(tail: int):
    """Delete cached shards older than `tail` blocks."""
    cur = await (await get_subtensor()).get_current_block()
    for f in CACHE_DIR.glob("*.jsonl"):
        b = f.stem.split("-", 1)[0]
        if b.isdigit() and int(b) < cur - tail: f.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
#                               QUERY                                         #
# --------------------------------------------------------------------------- #
HTTP_SEM = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
TERMINAL = {400, 404, 410}
async def query(prompt, model: str = "unsloth/gemma-3-12b-it", slug: str = "llm", timeout=150, retries=0, backoff=1) -> Response:
    url = f"https://{slug}.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
    start = time.monotonic()
    QCOUNT.labels(model=model).inc()
    R = lambda resp, at, err, ok: Response(response=resp, latency_seconds=time.monotonic()-start,
                                          attempts=at, model=model, error=err, success=ok)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        for attempt in range(1, retries+2):
            try:
                payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
                async with HTTP_SEM, sess.post(url, json=payload,
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
async def run(challenges, miners, timeout=150, retries=0, backoff=1 )-> List[Result]:
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
    # Fix: Get current event loop explicitly to prevent "Future attached to different loop" error
    loop = asyncio.get_running_loop()
    tasks = [ loop.create_task(proc(m, chal)) for m in mmap.values() if m.model for chal in challenges]  
    total = len(tasks); completed = 0
    
    # Fix: Use asyncio.gather with proper exception handling instead of as_completed
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                # Skip failed tasks but continue processing
                continue
            response.append(result)
            completed += 1
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
    except Exception as e:
        logger.error(f"Error in task execution: {e}")
        # Cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
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
            data = json.loads(data)
            model, miner_revision, chute_id = data.get("model"), data.get("revision"), data.get("chute_id")
            chute = await get_chute(chute_id)
            slug, chutes_revision = chute.get("slug"), chute.get("revision")
            if model.split('/')[1].lower()[:6] != 'affine': return None 
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
async def retry_set_weights( wallet: bt.Wallet, best_uid:int, retry: int = 10 ):
    for tries in range(retry):
        try:
            logger.info(f'Make subtensor connection...')
            sub = bt.subtensor( get_conf('SUBTENSOR_ENDPOINT', default='finney') )
            logger.info(f'Call set weights...')
            sub.set_weights(
                wallet = wallet,
                netuid=NETUID,
                weights=[1.0],
                uids=[best_uid],
                wait_for_inclusion=False
            )
            logger.info(f'Waiting 1 block ...')
            current_block = sub.get_current_block()
            await asyncio.sleep(12)
            logger.info(f'Checking last update...')
            meta = sub.metagraph(NETUID)
            last_update = meta.last_update[ meta.hotkeys.index( wallet.hotkey.ss58_address ) ]
            if last_update >= current_block:
                logger.info(f'Success, weight are on chain. Breaking loop.')
                return
            else:
                logger.warning(f"Failed transaction. Retrying {tries}/{retry} ...")
                continue
        except Exception as e:
            logger.warning(f'Error while setting weights: {e}, Retrying {tries}/{retry} ...')
            continue
        
TAIL= 10_000
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
        if hk not in cnt or not name.startswith('affine'):
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

    # select best
    best = max(prev, key=lambda hk: (dom[hk], -prev[hk].miner.block))
    best_uid = meta.hotkeys.index(best)

    # print summary
    hdr = ["UID","Model","Rev"] \
        + [f"{e}Acc" for e in ENVS] \
        + [f"{e}Rnk" for e in ENVS] \
        + [f"{e}N"   for e in ENVS] \
        + ["Dom","Wgt"]
    rows = sorted([
        [m.uid, m.model.split('/')[1][:20], m.revision[:5]]
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
    
@cli.command("weights")
@click.option('--tail','-t', default=TAIL, help="Results from tail blocks.")
def weights(tail:int): 
    """Computes current scores based on tail results."""
    asyncio.run(get_weights(tail=tail))
        
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
                x = (BLOCK % TEMPO != 0 or BLOCK <= LAST) and (TEMPO - (BLOCK % TEMPO)) if BLOCK > LAST else 0
                if BLOCK % TEMPO != 0 or BLOCK <= LAST: 
                    logger.info(f"Prefetching, {TEMPO-x}/{TEMPO} blocks until set weights ...")
                    await prefetch( TAIL )
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
