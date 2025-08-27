# --------------------------------------------------------------------------- #
#                             Postgres Integration                            #
# --------------------------------------------------------------------------- #
from __future__ import annotations

import os
import json
import asyncio
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, AsyncIterator

import aiofiles
import aiohttp
import orjson as _json
import asyncpg
from tqdm import tqdm

import bittensor as bt
import affine as af

from sqlalchemy import (
    Table, Column, MetaData, String, Integer, Float, Text, Boolean,
    DateTime, UniqueConstraint, Index, select, func
)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.engine import make_url

from .bucket import (
    get_client_ctx,        # S3/R2 client ctx manager (already configured)
    _loads, _dumps,        # fast json (orjson)
    RESULT_PREFIX as DEFAULT_RESULT_PREFIX,
    FOLDER,                # bucket name
)

import nest_asyncio
nest_asyncio.apply()

# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
from __future__ import annotations
import os
import json
import aiohttp
import asyncio
import aiofiles
from .utils import *
from tqdm import tqdm
import orjson as _json
import bittensor as bt
from pathlib import Path
from tqdm.asyncio import tqdm
from botocore.config import Config
from aiobotocore.session import get_session
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable, AsyncIterator

import affine as af

# --------------------------------------------------------------------------- #
#                             Dataset                                         #
# --------------------------------------------------------------------------- #

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
_loads, _dumps = _json.loads, _json.dumps
    
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
    async with aiofiles.open(p, "rb") as f:
        async for l in f: yield l.rstrip(b"\n")

# ── Core async stream (Result objects) ──────────────────────────────────────
async def rollouts(
    tail: int,
    *,
    max_concurrency: int = 10,      # parallel S3 downloads
) -> AsyncIterator["af.Result"]:
    """
    Stream `Result`s in deterministic order while pre‑downloading future
    shards concurrently.
    """
    # ── figure out which windows we need ────────────────────────────────
    sub  = await af.get_subtensor()
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
                r = af.Result.model_validate(_loads(raw))
                if r.verify():
                    bar.update(1)
                    yield r
            except Exception:
                pass
    bar.close()
    
    
# --------------------------------------------------------------------------- #
async def sign_results( wallet, results ):
    try:
        signer_url = af.get_conf('SIGNER_URL', default='http://signer:8080')
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
        af.logger.info(f"sink: signer unavailable, using local signing: {type(e).__name__}: {e}")
        hotkey = wallet.hotkey.ss58_address
        for r in results: 
            r.sign(wallet)
    finally:
        return hotkey, results

# ── Minimal sink / misc helpers (optional) ──────────────────────────────────
async def sink(wallet: bt.wallet, results: list["af.Result"], block: int = None):
    if not results: return
    if block is None:
        sub = await af.get_subtensor(); block = await sub.get_current_block()
    hotkey, signed = await sign_results( wallet, results )
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


# --------------------------------------------------------------------------- #
#                          Config / Constants                                 #
# --------------------------------------------------------------------------- #

WINDOW              = int(os.getenv("AFFINE_WINDOW", "20"))
CACHE_DIR           = Path(os.getenv("AFFINE_CACHE_DIR",
                             Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Runners write shards here; populate() reads from here
RESULTS_PREFIX      = os.getenv("RESULT_PREFIX", DEFAULT_RESULT_PREFIX).rstrip("/") + "/"

# Index (V2 supports {key,lm} entries; V1 is list[str] of keys)
INDEX_KEY           = os.getenv("R2_INDEX_KEY", "affine/index.json")
TIMEZONE            = dt.timezone.utc

DATABASE_URL        = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/affine")
VERIFY_SIGNATURE    = os.getenv("VERIFY_SIGNATURE", "0").lower() in {"1", "true", "yes"}
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "1000"))
LIST_PAGE_SIZE      = int(os.getenv("R2_LIST_PAGE_SIZE", "1000"))

logger = getattr(af, "logger", None) or __import__("logging").getLogger(__name__)

def _w(b: int) -> int:
    return (b // WINDOW) * WINDOW


# --------------------------------------------------------------------------- #
#                              Tables / Schema                                #
# --------------------------------------------------------------------------- #

metadata = MetaData()

affine_results = Table(
    "affine_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),

    # Flat columns
    Column("env_name", String(128), nullable=False),
    Column("env_version", String(32), nullable=False),
    Column("uid", Integer, nullable=False),
    Column("hotkey", String(128), nullable=False),
    Column("model", String(512), nullable=True),
    Column("revision", String(128), nullable=True),
    Column("prompt", Text, nullable=True),
    Column("response", Text, nullable=True),
    Column("score", Float, nullable=True),

    # Ops / idempotency
    Column("challenge_id", String(64), nullable=False),
    Column("success", Boolean, nullable=True),
    Column("latency_seconds", Float, nullable=True),
    Column("attempts", Integer, nullable=True),
    Column("error", Text, nullable=True),
    Column("miner_slug", String(256), nullable=True),
    Column("miner_block", Integer, nullable=True),
    Column("result_version", String(32), nullable=True),
    Column("signer_hotkey", String(128), nullable=True),

    # Provenance & ingestion times
    Column("r2_key", String(512), nullable=False),
    Column("r2_last_modified", DateTime(timezone=True), nullable=False),
    Column("ingested_at", DateTime(timezone=True), server_default=func.now(), nullable=False),

    # Future multi-turn
    Column("conversation_id", String(128), nullable=True),
    Column("turn_index", Integer, nullable=True),
    Column("message_index", Integer, nullable=True),
    Column("role", String(32), nullable=True),
    Column("extra", JSONB, nullable=True),

    UniqueConstraint("hotkey", "challenge_id", name="uq_hotkey_challenge"),
)

Index("ix_results_env", affine_results.c.env_name)
Index("ix_results_hotkey", affine_results.c.hotkey)
Index("ix_results_r2lm", affine_results.c.r2_last_modified.desc())

ingest_state = Table(
    "affine_ingest_state",
    metadata,
    Column("state_id", String(32), primary_key=True),  # 'r2'
    Column("last_key", String(512), nullable=True),
    Column("last_modified", DateTime(timezone=True), nullable=True),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

# --------------------------------------------------------------------------- #
#                             Engine / Sessions                               #
# --------------------------------------------------------------------------- #

_engine: Optional[Any] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None

async def _ensure_database_exists(url_str: str) -> None:
    """Best-effort create target DB if missing."""
    url = make_url(url_str)
    if not url.get_backend_name().endswith("+asyncpg"):
        return
    host = url.host or "localhost"
    user = url.username or "postgres"
    password = url.password or ""
    port = url.port or 5432
    dbname = url.database or "affine"

    try:
        conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=dbname)
        await conn.close()
        return
    except asyncpg.InvalidCatalogNameError:
        pass
    except Exception:
        pass  # keep trying

    admin_db = "postgres"
    conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=admin_db)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", dbname)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        await conn.close()

async def _get_engine():
    global _engine, _sessionmaker
    if _engine is None:
        await _ensure_database_exists(DATABASE_URL)
        _engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
        _sessionmaker = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
        async with _engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
    return _engine

def _sm() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("DB not initialized; call populate() or stream() once to init.")
    return _sessionmaker

# --------------------------------------------------------------------------- #
#                             Index Helpers                                   #
# --------------------------------------------------------------------------- #
# V2 index format: [{"key": ".../000000100-hotkey.json", "lm": "2025-08-27T11:23:45.123456+00:00"}, ...]
# V1 index format: [".../000000100-hotkey.json", ...]  (no last-modifieds)

async def _read_index_entries() -> Optional[List[Tuple[str, dt.datetime]]]:
    """Return list of (key, last_modified). Supports V2 (preferred) and V1 (fallback)."""
    try:
        async with get_client_ctx() as c:
            r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
            body = await r["Body"].read()
    except Exception:
        return None

    try:
        data = json.loads(body)
    except Exception:
        return None

    entries: List[Tuple[str, dt.datetime]] = []
    if not data:
        return []

    # V2: list of dicts
    if isinstance(data, list) and data and isinstance(data[0], dict):
        for d in data:
            key = d.get("key")
            lm = d.get("lm")
            if key and lm:
                try:
                    when = dt.datetime.fromisoformat(lm)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=TIMEZONE)
                    entries.append((key, when))
                except Exception:
                    continue
        return entries

    # V1: list of keys -> we don’t know LM; return None to force list_objects fallback
    if isinstance(data, list) and (not data or isinstance(data[0], str)):
        return None

    return None

async def _update_index_v2(key: str, lm: dt.datetime) -> None:
    """Idempotently add {key,lm} to V2 index, sorted by (lm,key)."""
    try:
        async with get_client_ctx() as c:
            try:
                r = await c.get_object(Bucket=FOLDER, Key=INDEX_KEY)
                existing = json.loads(await r["Body"].read())
            except c.exceptions.NoSuchKey:
                existing = []
            # Dedup: replace if key exists with older lm
            best = {}
            for e in existing:
                if isinstance(e, dict) and "key" in e and "lm" in e:
                    best[e["key"]] = e["lm"]
            new_lm = lm.astimezone(dt.timezone.utc).isoformat()
            prev = best.get(key)
            if (prev is None) or (prev < new_lm):
                best[key] = new_lm
            out = [{"key": k, "lm": best[k]} for k in best]
            out.sort(key=lambda x: (x["lm"], x["key"]))
            await c.put_object(Bucket=FOLDER, Key=INDEX_KEY,
                               Body=_dumps(out), ContentType="application/json")
    except Exception as e:
        logger.debug(f"index v2 update failed (non-fatal): {e}")

# --------------------------------------------------------------------------- #
#                          Result <-> Row Mapping                             #
# --------------------------------------------------------------------------- #

def _result_to_row(r: "af.Result", r2_key: str, r2_last_modified: dt.datetime) -> Dict[str, Any]:
    env_obj = getattr(r.challenge, "env", None)
    env_name = getattr(env_obj, "name", str(env_obj))
    env_version = getattr(env_obj, "__version__", "0.0.0")

    challenge_id = getattr(r.challenge, "challenge_id", None)
    prompt = getattr(r.challenge, "prompt", None)
    evaluation = getattr(r, "evaluation", None)
    response_obj = getattr(r, "response", None)

    response_text = getattr(response_obj, "response", None) if response_obj else None
    score = getattr(evaluation, "score", None) if evaluation else None

    miner = getattr(r, "miner", None)
    uid = getattr(miner, "uid", None)
    hotkey = getattr(miner, "hotkey", None)
    model = getattr(miner, "model", None)
    revision = getattr(miner, "revision", None)
    miner_slug = getattr(miner, "slug", None)
    miner_block = getattr(miner, "block", None)

    success = getattr(response_obj, "success", None) if response_obj else None
    latency = getattr(response_obj, "latency_seconds", None) if response_obj else None
    attempts = getattr(response_obj, "attempts", None) if response_obj else None
    error = getattr(response_obj, "error", None) if response_obj else None

    signer_hotkey = getattr(r, "hotkey", None)
    result_version = getattr(r, "version", None)

    extra = {
        "challenge_extra": getattr(r.challenge, "extra", None),
        "evaluation_extra": getattr(evaluation, "extra", None) if evaluation else None,
        "miner_chute": getattr(miner, "chute", None),
    }

    return {
        "env_name": env_name,
        "env_version": env_version,
        "uid": uid,
        "hotkey": hotkey,
        "model": model,
        "revision": revision,
        "prompt": prompt,
        "response": response_text,
        "score": score,

        "challenge_id": challenge_id,
        "success": success,
        "latency_seconds": latency,
        "attempts": attempts,
        "error": error,
        "miner_slug": miner_slug,
        "miner_block": int(miner_block) if isinstance(miner_block, int) or (isinstance(miner_block, str) and miner_block.isdigit()) else None,
        "result_version": result_version,
        "signer_hotkey": signer_hotkey,

        "r2_key": r2_key,
        "r2_last_modified": r2_last_modified,
        "extra": extra,

        "conversation_id": None,
        "turn_index": None,
        "message_index": None,
        "role": "assistant" if success else None,
    }

# --------------------------------------------------------------------------- #
#                            Runner-Side Sink                                 #
# --------------------------------------------------------------------------- #

async def sign_results(wallet, results: List["af.Result"]) -> Tuple[str, List["af.Result"]]:
    try:
        signer_url = af.get_conf("SIGNER_URL", default="http://signer:8080")
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
                return hotkey, results
    except Exception as e:
        logger.info(f"sink: signer unavailable, using local signing: {type(e).__name__}: {e}")
        pass

    hotkey = wallet.hotkey.ss58_address
    for r in results:
        r.sign(wallet)
    return hotkey, results

async def sink(wallet: bt.wallet, results: List["af.Result"], block: Optional[int] = None):
    """Runner pushes results to R2; updates V2 index with (key,lm)."""
    if not results:
        return
    if block is None:
        sub = await af.get_subtensor()
        block = await sub.get_current_block()

    hotkey, signed = await sign_results(wallet, results)
    key = f"{RESULTS_PREFIX}{_w(block):09d}-{hotkey}.json"
    dumped = [r.model_dump(mode="json") for r in signed]

    async with get_client_ctx() as c:
        try:
            r = await c.get_object(Bucket=FOLDER, Key=key)
            merged = json.loads(await r["Body"].read()) + dumped
        except Exception:
            merged = dumped

        await c.put_object(Bucket=FOLDER, Key=key, Body=_dumps(merged), ContentType="application/json")

        # Get authoritative LM and update V2 index
        try:
            h = await c.head_object(Bucket=FOLDER, Key=key)
            lm = h["LastModified"]
            if isinstance(lm, dt.datetime) and lm.tzinfo is None:
                lm = lm.replace(tzinfo=TIMEZONE)
            await _update_index_v2(key, lm)
        except Exception as e:
            logger.debug(f"index update skipped: {e}")

# --------------------------------------------------------------------------- #
#                       Populate (R2 -> Postgres)                              #
# --------------------------------------------------------------------------- #

async def _load_state(session: AsyncSession) -> Dict[str, Any]:
    res = await session.execute(
        select(ingest_state.c.last_key, ingest_state.c.last_modified)
        .where(ingest_state.c.state_id == "r2")
        .limit(1)
    )
    row = res.first()
    if row:
        last_key, last_modified = row
        return {"last_key": last_key, "last_modified": last_modified}
    return {"last_key": None, "last_modified": None}

async def _save_state(session: AsyncSession, last_key: str, last_modified: dt.datetime) -> None:
    stmt = pg_insert(ingest_state).values(
        state_id="r2", last_key=last_key, last_modified=last_modified
    ).on_conflict_do_update(
        index_elements=[ingest_state.c.state_id],
        set_={"last_key": last_key, "last_modified": last_modified, "updated_at": func.now()},
    )
    await session.execute(stmt)
    await session.commit()

async def _list_objects_since_via_index(
    since: Optional[dt.datetime], last_key: Optional[str]
) -> Optional[List[Dict[str, Any]]]:
    """
    Use V2 index if available. Returns [{'Key': str, 'LastModified': dt}, ...] or None.
    Filters by (LastModified, Key) > (since, last_key).
    """
    entries = await _read_index_entries()
    if entries is None:
        return None
    objs: List[Dict[str, Any]] = []
    for key, lm in entries:
        if since is None:
            objs.append({"Key": key, "LastModified": lm})
        else:
            if (lm > since) or (lm == since and last_key and key > last_key):
                objs.append({"Key": key, "LastModified": lm})
    objs.sort(key=lambda x: (x["LastModified"], x["Key"]))
    return objs

async def _list_objects_since_fallback(
    since: Optional[dt.datetime], last_key: Optional[str]
) -> List[Dict[str, Any]]:
    """Fallback to S3 listing by prefix, watermark-aware."""
    objs: List[Dict[str, Any]] = []
    async with get_client_ctx() as c:
        paginator = c.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=FOLDER, Prefix=RESULTS_PREFIX, PaginationConfig={"PageSize": LIST_PAGE_SIZE}
        ):
            for o in page.get("Contents", []):
                key = o.get("Key", "")
                if not key.endswith(".json"):
                    continue
                lm = o.get("LastModified")
                if isinstance(lm, dt.datetime) and lm.tzinfo is None:
                    lm = lm.replace(tzinfo=TIMEZONE)
                add = False
                if since is None:
                    add = True
                else:
                    if (lm > since) or (lm == since and last_key and key > last_key):
                        add = True
                if add:
                    objs.append({"Key": key, "LastModified": lm})
    objs.sort(key=lambda x: (x["LastModified"], x["Key"]))
    return objs

async def _read_r2_json_array(key: str) -> List[bytes]:
    async with get_client_ctx() as c:
        o = await c.get_object(Bucket=FOLDER, Key=key)
        body = await o["Body"].read()
    arr = _loads(body)
    if isinstance(arr, list):
        return [_dumps(item) for item in arr]
    return []

async def _pull_once(verify_signature: bool, override_since: Optional[dt.datetime]) -> int:
    """
    Incremental ingest from R2 into Postgres using watermark & index V2 when available.
    Returns count of rows attempted (duplicates skipped by ON CONFLICT).
    """
    await _get_engine()
    sm = _sm()

    attempted = 0
    async with sm() as session:
        state = await _load_state(session)
        since = override_since or state["last_modified"]
        last_key = None if override_since else state["last_key"]

        logger.info(f"[populate] Using watermark since={since} last_key={last_key} prefix={RESULTS_PREFIX}")

        objs = await _list_objects_since_via_index(since, last_key)  # try V2 index
        if objs is None:
            logger.info("[populate] Index V2 not found or legacy format; falling back to bucket listing.")
            objs = await _list_objects_since_fallback(since, last_key)

        if not objs:
            logger.info("[populate] No new objects to process.")
            return 0

        logger.info(f"[populate] Processing {len(objs)} objects")

        # File-level progress bar
        file_bar = tqdm(total=len(objs), desc="Files", unit="file", dynamic_ncols=True, leave=False)
        row_bar = tqdm(total=0, desc="Rows (this run)", unit="row", dynamic_ncols=True, leave=True)

        for i, obj in enumerate(objs, 1):
            key = obj["Key"]
            lm = obj["LastModified"]
            logger.info(f"[populate] {i}/{len(objs)} {key} (lm={lm.isoformat() if hasattr(lm,'isoformat') else lm})")

            try:
                raw_blobs = await _read_r2_json_array(key)
            except Exception as e:
                logger.warning(f"[populate] read error: {key}: {e}")
                file_bar.update(1)
                continue

            rows: List[Dict[str, Any]] = []
            valid, bad_sig, parse_err = 0, 0, 0

            for raw in raw_blobs:
                try:
                    d = _loads(raw)
                    r = af.Result.model_validate(d)
                    if verify_signature and not r.verify():
                        bad_sig += 1
                        continue
                    rows.append(_result_to_row(r, key, lm))
                    valid += 1
                except Exception:
                    parse_err += 1
                    continue

                if len(rows) >= BATCH_SIZE:
                    stmt = pg_insert(affine_results).values(rows)
                    stmt = stmt.on_conflict_do_nothing(index_elements=["hotkey", "challenge_id"])
                    await session.execute(stmt)
                    await session.commit()
                    attempted += len(rows)
                    row_bar.total += len(rows)
                    row_bar.update(len(rows))
                    rows.clear()

            if rows:
                stmt = pg_insert(affine_results).values(rows)
                stmt = stmt.on_conflict_do_nothing(index_elements=["hotkey", "challenge_id"])
                await session.execute(stmt)
                await session.commit()
                attempted += len(rows)
                row_bar.total += len(rows)
                row_bar.update(len(rows))

            logger.info(f"[populate] done {key}: valid={valid} sigfail={bad_sig} parseerr={parse_err}")

            # Advance watermark *after* this object completes
            await _save_state(session, last_key=key, last_modified=lm)

            file_bar.update(1)

        file_bar.close()
        row_bar.close()

    logger.info(f"[populate] completed: attempted={attempted} (duplicates skipped)")
    return attempted

def populate(since: Optional[dt.datetime] = None, verify_signature: Optional[bool] = None) -> int:
    """
    Pull latest data from R2 and populate Postgres.
      - since: override watermark (tz-aware). None = continue from saved watermark.
      - verify_signature: True => require Result.verify() to pass.
    """
    vs = VERIFY_SIGNATURE if verify_signature is None else bool(verify_signature)
    return asyncio.run(_pull_once(verify_signature=vs, override_since=since))

# --------------------------------------------------------------------------- #
#                               Query (stream)                                #
# --------------------------------------------------------------------------- #

async def _stream_fetch(
    since: Optional[dt.datetime] = None,
    env_name: Optional[str] = None,
    uid: Optional[int] = None,
    model: Optional[str] = None,
    limit: int = 1000,
    order: str = "r2_last_modified",
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    await _get_engine()
    sm = _sm()

    cols = [
        affine_results.c.env_name,
        affine_results.c.env_version,
        affine_results.c.uid,
        affine_results.c.hotkey,
        affine_results.c.model,
        affine_results.c.revision,
        affine_results.c.prompt,
        affine_results.c.response,
        affine_results.c.score,
        affine_results.c.r2_last_modified,
    ]
    stmt = select(*cols)

    if since is not None:
        stmt = stmt.where(affine_results.c.r2_last_modified >= since)
    if env_name:
        stmt = stmt.where(affine_results.c.env_name == env_name)
    if uid is not None:
        stmt = stmt.where(affine_results.c.uid == uid)
    if model:
        stmt = stmt.where(affine_results.c.model == model)

    if order == "r2_last_modified":
        ob = affine_results.c.r2_last_modified.asc() if ascending else affine_results.c.r2_last_modified.desc()
    elif order == "score":
        ob = affine_results.c.score.asc() if ascending else affine_results.c.score.desc()
    else:
        ob = affine_results.c.id.asc() if ascending else affine_results.c.id.desc()

    stmt = stmt.order_by(ob).limit(limit)

    async with sm() as session:
        res = await session.execute(stmt)
        # SQLAlchemy Row -> dict (py39+)
        return [dict(r._mapping) for r in res.fetchall()]

def stream(
    since: Optional[dt.datetime] = None,
    env_name: Optional[str] = None,
    uid: Optional[int] = None,
    model: Optional[str] = None,
    limit: int = 1000,
    order: str = "r2_last_modified",
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """
    Query ingested results (flat rows).
      - Filters: since (tz-aware), env_name, uid, model
      - order: 'r2_last_modified' | 'score' | 'id'
    """
    return asyncio.run(_stream_fetch(since, env_name, uid, model, limit, order, ascending))
