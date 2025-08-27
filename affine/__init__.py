
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

from .logging import *

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
            

from .envs import *
from .signer import *
from .database import *
from .chutes import *
from .runner import *
from .validator import *
from .miners import * 

# --------------------------------------------------------------------------- #
#                          Dataset upload CLI                                 #
# --------------------------------------------------------------------------- #
import click
import datasets as hf_ds
import asyncio as _asyncio
from sqlalchemy.dialects.postgresql import insert as _pg_insert
from sqlalchemy.exc import DBAPIError
from .database import _get_engine as _get_engine, _sm as _sm, dataset_rows as dataset_rows
import asyncpg

@cli.command("upload-dataset")
@click.argument("dataset_name", type=str)
@click.option("--config", "config", type=str, default="default", help="Dataset config (HF)")
@click.option("--split", "split", type=str, default="train", help="Dataset split (HF)")
def upload_dataset_cmd(dataset_name: str, config: str, split: str):
    """Upload an HF dataset's rows into Postgres `dataset_rows`.

    Example:
      affine upload-dataset satpalsr/rl-python --config default --split train
    """
    async def _run():
        af.logger.debug(f"Starting upload for dataset: {dataset_name} with config: {config} and split: {split}")
        await _get_engine()
        sm = _sm()
        ds = hf_ds.load_dataset(dataset_name, name=None if config == "default" else config, split=split)
        af.logger.debug(f"Loaded dataset: {dataset_name} with {len(ds)} rows")
        batch: list[dict] = []
        BATCH = BATCH_SIZE
        idx = 0
        total_rows = len(ds)
        async with sm() as session:
            def _make_stmt(rows: list[dict]):
                af.logger.debug(f"Preparing statement for batch of size: {len(rows)}")
                values = [
                    {
                        "dataset_name": dataset_name,
                        "config": config,
                        "split": split,
                        "row_index": r["__row_index__"],
                        "data": {k: v for k, v in r.items() if k != "__row_index__"},
                    }
                    for r in rows
                ]
                stmt = _pg_insert(dataset_rows).values(values)
                # Idempotent upsert: on conflict, do nothing
                stmt = stmt.on_conflict_do_nothing(index_elements=[
                    dataset_rows.c.dataset_name,
                    dataset_rows.c.config,
                    dataset_rows.c.split,
                    dataset_rows.c.row_index,
                ])
                return stmt

            async def _execute_batch_with_retries(rows: list[dict], *, max_retries: int = 5) -> None:
                """Execute and commit a batch with retries on transient disconnects."""
                delay = 0.5
                attempt = 0
                while True:
                    try:
                        af.logger.debug(f"Executing batch of size: {len(rows)} (attempt {attempt+1})")
                        await session.execute(_make_stmt(rows))
                        await session.commit()
                        af.logger.debug("Batch committed")
                        return
                    except (DBAPIError, asyncpg.ConnectionDoesNotExistError) as e:
                        try:
                            await session.rollback()
                        except Exception:
                            pass
                        is_invalidated = isinstance(e, DBAPIError) and getattr(e, "connection_invalidated", False)
                        msg = str(getattr(e, "orig", e)).lower()
                        is_disconnect = isinstance(e, asyncpg.ConnectionDoesNotExistError) or "connection was closed" in msg
                        retriable = is_invalidated or is_disconnect
                        attempt += 1
                        if not retriable or attempt >= max_retries:
                            af.logger.error(f"Giving up on batch after {attempt} attempts due to error: {e}")
                            raise
                        af.logger.warning(f"Transient DB disconnect during batch (attempt {attempt}); retrying in {delay:.1f}s…")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 5.0)

            # Iterate rows
            for row in ds:  # type: ignore
                # Attach running index; if HF provides _keys, we still assign our own index
                payload = dict(row)
                payload["__row_index__"] = idx
                batch.append(payload)
                idx += 1
                if len(batch) >= BATCH:
                    await _execute_batch_with_retries(batch)
                    batch.clear()
                # Show progress
                af.logger.info(f"Progress: {idx}/{total_rows} rows uploaded")

            if batch:
                af.logger.debug(f"Executing final batch of size: {len(batch)}")
                await _execute_batch_with_retries(batch)
                af.logger.debug(f"Final batch committed")
                af.logger.info(f"Progress: {idx}/{total_rows} rows uploaded")

        af.logger.info(f"Uploaded {idx} rows for {dataset_name} [{config}/{split}] to dataset_rows")

    _asyncio.run(_run())
