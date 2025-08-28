from __future__ import annotations
import os
import click
import json
import aiohttp
import asyncio
import asyncpg
import nest_asyncio
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.exc import OperationalError, ProgrammingError, DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.engine import make_url
from sqlalchemy import (
    Table, Column, MetaData, String, Integer, Float, Text, Boolean,
    DateTime, UniqueConstraint, Index, select, func
)
import affine as af
nest_asyncio.apply()

TIMEZONE            = dt.timezone.utc
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "1000"))
USERNAME            =  "app_reader" #"postgres"
r2_key = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
if r2_key:
    PASSWORD = r2_key[:14]
    USERNAME = "writer_user2"
else:
    PASSWORD = "ca35a0d8bd31d0d5"
    USERNAME = "app_reader"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{USERNAME}:{PASSWORD}@database-1.clo608s4ivev.us-east-1.rds.amazonaws.com:5432/postgres"
)

# --------------------------------------------------------------------------- #
#                         Postgres schema / engine                             #
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

    # Provenance
    Column("r2_key", String(512), nullable=False),
    Column("r2_last_modified", DateTime(timezone=True), nullable=False),
    Column("ingested_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
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

# --------------------------------------------------------------------------- #
#                         Dataset storage (minimal)                           #
# --------------------------------------------------------------------------- #
# Stores raw dataset rows in JSONB, keyed by (dataset_name, config, split, row_index)
dataset_rows = Table(
    "dataset_rows",
    metadata,
    Column("dataset_name", String(255), primary_key=True, nullable=False),
    Column("config", String(255), primary_key=True, nullable=False),
    Column("split", String(128), primary_key=True, nullable=False),
    Column("row_index", Integer, primary_key=True, nullable=False),
    Column("data", JSONB, nullable=False),
    Column("ingested_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

Index("ix_ds_name", dataset_rows.c.dataset_name)
Index("ix_ds_ns", dataset_rows.c.dataset_name, dataset_rows.c.config, dataset_rows.c.split, dataset_rows.c.row_index)

_engine: Optional[Any] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None

async def _ensure_database_exists(url_str: str) -> None:
    """Best-effort: create target DB if missing (requires create permission)."""
    url = make_url(url_str)
    backend = url.get_backend_name() or "postgresql+asyncpg"
    if not backend.startswith("postgresql"):
        return  # only Postgres is supported for auto-create
    host = url.host or "localhost"
    user = url.username or "postgres"
    password = url.password or ""
    port = url.port or 5432
    dbname = url.database or "affine"

    # Attempt initial connection
    try:
        conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=dbname, ssl="require")
        await conn.close()
        return
    except asyncpg.InvalidCatalogNameError:
        pass  # reachable but DB missing; proceed to create
    except Exception:
        # Do not attempt to start a local Dockerized Postgres; proceed to admin DB checks
        pass

    # Connect to an admin database and create the target DB if missing
    admin_db_candidates = ["postgres", "template1"]
    admin_conn = None
    last_err: Exception | None = None
    for admin_db in admin_db_candidates:
        try:
            admin_conn = await asyncpg.connect(host=host, port=port, user=user, password=password, database=admin_db, ssl="require")
            break
        except Exception as e:
            last_err = e
            continue
    if admin_conn is None:
        if last_err:
            raise last_err
        return

    try:
        exists = await admin_conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", dbname)
        if not exists:
            try:
                await admin_conn.execute(f'CREATE DATABASE "{dbname}"')
            except Exception:
                # If another process created it concurrently or permission issues, we'll verify below
                pass
    finally:
        await admin_conn.close()

    # Verify the database is connectable; retry briefly for safety
    for _ in range(20):
        try:
            test = await asyncpg.connect(host=host, port=port, user=user, password=password, database=dbname, ssl="require")
            await test.close()
            break
        except asyncpg.InvalidCatalogNameError:
            await asyncio.sleep(0.5)
        except Exception:
            await asyncio.sleep(0.5)

async def _get_engine():
    global _engine, _sessionmaker
    if _engine is None:
        await _ensure_database_exists(DATABASE_URL)
        # Connection pool sizing (configurable via env):
        # - POOL_SIZE: number of persistent connections in pool
        # - MAX_OVERFLOW: extra transient connections beyond pool_size
        # - POOL_TIMEOUT: seconds to wait for connection checkout
        # Conservative defaults to avoid exhausting Postgres when many runners are active
        pool_size = int(os.getenv("DB_POOL_SIZE", os.getenv("POOL_SIZE", "4")))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", os.getenv("MAX_OVERFLOW", "2")))
        pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", os.getenv("POOL_TIMEOUT", "60")))
        _engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            connect_args={"ssl": "require"},
            pool_pre_ping=True,
            pool_recycle=180,
            pool_timeout=pool_timeout,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_use_lifo=True,
        )
        _sessionmaker = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with _engine.begin() as conn:
                await conn.run_sync(metadata.create_all)
        except (asyncpg.InvalidCatalogNameError, OperationalError, ProgrammingError, DBAPIError) as e:
            # Handle missing database cases raised via different exception wrappers
            msg = str(getattr(e, "orig", e)).lower()
            if ("invalidcatalogname" in msg) or ("does not exist" in msg and "database" in msg):
                await _ensure_database_exists(DATABASE_URL)
                await _engine.dispose()
                _engine = create_async_engine(
                    DATABASE_URL,
                    echo=False,
                    connect_args={"ssl": "require"},
                    pool_pre_ping=True,
                    pool_recycle=180,
                    pool_timeout=pool_timeout,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_use_lifo=True,
                )
                _sessionmaker = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
                await asyncio.sleep(0.5)
                async with _engine.begin() as conn:
                    await conn.run_sync(metadata.create_all)
            else:
                raise
    return _engine

def _sm() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("DB not initialized; call populate() or stream() once to init.")
    return _sessionmaker


# --------------------------------------------------------------------------- #
#               Dataset helpers: insert/select from dataset_rows               #
# --------------------------------------------------------------------------- #
async def select_dataset_rows(
    *,
    dataset_name: str,
    config: str = "default",
    split: str = "train",
    limit: int = 1000,
    offset: int = 0,
    include_index: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch rows from `dataset_rows` ordered by row_index.

    If include_index is False (default), returns only the JSON payloads.
    If True, returns dicts with keys: row_index, data.
    """
    await _get_engine()
    sm = _sm()
    stmt = (
        select(
            dataset_rows.c.row_index if include_index else dataset_rows.c.data
        )
        .where(dataset_rows.c.dataset_name == dataset_name)
        .where(dataset_rows.c.config == config)
        .where(dataset_rows.c.split == split)
        .order_by(dataset_rows.c.row_index.asc())
        .offset(offset)
        .limit(limit)
    )
    async with sm() as session:
        res = await session.execute(stmt)
        out = []
        for row in res.fetchall():
            if include_index:
                m = row._mapping
                out.append({
                    "row_index": int(m.get("row_index")),
                    "data": m.get("data"),
                })
            else:
                val = row[0]
                out.append(val)
        return out


# --------------------------------------------------------------------------- #
#               Query helper: generic count with flexible filters             #
# --------------------------------------------------------------------------- #
async def count(**filters: Any) -> int:
    """
    Count rows in `affine_results` matching arbitrary column filters.

    Examples:
        await count(env="DED", hotkey="...", success=True)
        await count(revision="abc123", uid=5)
        await count(env=["DED", "ABD"], success=True)

    Notes:
        - `env` is an alias for `env_name`.
        - If a filter value is a list/tuple/set, an IN(...) predicate is used.
        - If a filter value is None, IS NULL is used.
    """
    await _get_engine()
    sm = _sm()

    alias = {"env": "env_name"}

    def _resolve_column(name: str):
        col_name = alias.get(name, name)
        col = getattr(affine_results.c, col_name, None)
        if col is None:
            raise ValueError(f"Unknown column filter: {name}")
        return col

    stmt = select(func.count()).select_from(affine_results)

    for key, value in filters.items():
        col = _resolve_column(key)
        if isinstance(value, (list, tuple, set)):
            stmt = stmt.where(col.in_(list(value)))
        elif value is None:
            stmt = stmt.where(col.is_(None))
        else:
            stmt = stmt.where(col == value)

    async with sm() as session:
        total = await session.scalar(stmt)
        return int(total or 0)

# ------------------------------------------------------------ #
#               Query helper: generic select with flexible filters            #
# --------------------------------------------------------------------------- #
async def select_rows(
    *,
    limit: int = 1000,
    order: str = "r2_last_modified",
    ascending: bool = False,
    **filters: Any,
) -> List[Dict[str, Any]]:
    """
    Return rows from `affine_results` matching arbitrary filters.

    Examples:
        await select_rows(env="DED", hotkey="...", success=True)
        await select_rows(revision="abc123", uid=5, limit=100)
        await select_rows(env=["DED", "ABD"], success=True, order="score")

    Notes:
        - `env` is an alias for `env_name`.
        - If a filter value is a list/tuple/set, an IN(...) predicate is used.
        - If a filter value is None, IS NULL is used.
        - order: 'r2_last_modified' | 'score' | 'id'
    """
    await _get_engine()
    sm = _sm()
    alias = {"env": "env_name"}
    def _resolve_column(name: str):
        col_name = alias.get(name, name)
        col = getattr(affine_results.c, col_name, None)
        if col is None:
            raise ValueError(f"Unknown column filter: {name}")
        return col
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
        affine_results.c.success,
        affine_results.c.miner_block,
        affine_results.c.r2_last_modified,
    ]
    stmt = select(*cols)
    for key, value in filters.items():
        col = _resolve_column(key)
        if isinstance(value, (list, tuple, set)):
            stmt = stmt.where(col.in_(list(value)))
        elif value is None:
            stmt = stmt.where(col.is_(None))
        else:
            stmt = stmt.where(col == value)
    if order == "r2_last_modified":
        ob = affine_results.c.r2_last_modified.asc() if ascending else affine_results.c.r2_last_modified.desc()
    elif order == "score":
        ob = affine_results.c.score.asc() if ascending else affine_results.c.score.desc()
    else:
        ob = affine_results.c.id.asc() if ascending else affine_results.c.id.desc()
    stmt = stmt.order_by(ob).limit(limit)
    async with sm() as session:
        res = await session.execute(stmt)
        return [dict(row._mapping) for row in res.fetchall()]
    

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
        "miner_block": int(miner_block) if isinstance(miner_block, int) or (isinstance(miner_block, str) and str(miner_block).isdigit()) else None,
        "result_version": result_version,
        "signer_hotkey": signer_hotkey,
        "r2_key": r2_key,
        "r2_last_modified": r2_last_modified,
        "conversation_id": None,
        "turn_index": None,
        "message_index": None,
        "role": "assistant" if success else None,
        "extra": extra,
    }

# --------------------------------------------------------------------------- #
#                         Direct DB signing and sink                           #
# --------------------------------------------------------------------------- #
async def sink(*, wallet: Any, results: List["af.Result"], block: Optional[int] = None) -> None:
    """Persist results directly into Postgres.

    - Signs results first (remote signer preferred, local fallback)
    - Uses (hotkey, challenge_id) ON CONFLICT DO NOTHING to ensure idempotency
    - Fills r2_key and r2_last_modified with synthetic values for schema compatibility
    """
    if not results:return
    hotkey_addr, signed = await af.sign_results(wallet, results)
    synthetic_key = f"direct/{block}-{hotkey_addr}"
    now_ts = dt.datetime.now(tz=TIMEZONE)
    rows: List[Dict[str, Any]] = [
        _result_to_row(r, synthetic_key, now_ts) for r in signed
    ]
    await _get_engine()
    sm = _sm()
    async with sm() as session:
        # Insert in batches for large lists
        for i in range(0, len(rows), BATCH_SIZE):
            chunk = rows[i : i + BATCH_SIZE]
            stmt = pg_insert(affine_results).values(chunk)
            stmt = stmt.on_conflict_do_nothing(index_elements=["hotkey", "challenge_id"])
            await session.execute(stmt)
            await session.commit()