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
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)
    
# --------------------------------------------------------------------------- #
#                   Env builder (AFFINE_ENV_LIST support)                     #
# --------------------------------------------------------------------------- #
def _build_envs() -> List[BaseEnv]:
    """Build active envs from AFFINE_ENV_LIST; accept bare names as agentgym."""
    spec = os.getenv("AFFINE_ENV_LIST", "").strip()
    if not spec:
        raise RuntimeError("AFFINE_ENV_LIST is required and must list envs, e.g. 'webshop,agentgym:alfworld,affine:sat'")
    envs: List[BaseEnv] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if ":" in tok:
            prefix, name = tok.split(":", 1)
            if prefix == "agentgym":
                envs.append(AgentGymContainerEnv(env_name=name))
            elif prefix == "affine":
                envs.append(AffineContainerEnv(env_name=name))
            else:
                logger.warning(f"Unknown env prefix in AFFINE_ENV_LIST: {prefix}")
        else:
            envs.append(AgentGymContainerEnv(env_name=tok))
    if not envs:
        raise RuntimeError("AFFINE_ENV_LIST contained no supported entries.")
    return envs

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
        envs = _build_envs()

        # ── config ───────────────────────────────────────────────────────────
        MAX_USES       = 30
        REFRESH_S      = 600     # metagraph/miners refresh cadence (s)
        SINK_BATCH     = 300     # flush threshold
        SINK_MAX_WAIT  = 60*5      # max seconds to hold partial batch
        BACKOFF0       = 5
        BACKOFF_CAP    = 300

        # ── state ───────────────────────────────────────────────────────────
        # Challenge cache per env (reused as placeholder to keep stable challenge_id)
        chal_cache: Dict[str, Tuple[Challenge, int]] = {}
        last_sync = 0.0
        miners_map: Dict[int, Miner] = {}
        # Per-env rounds and inflight tasks
        env_round: Dict[str, int] = {e.name: 0 for e in envs}
        env_inflight: Dict[str, Dict[int, asyncio.Task]] = {e.name: {} for e in envs}

        # results pipeline
        sink_q: asyncio.Queue = asyncio.Queue()

        # monitoring state
        last_status_log = 0.0
        total_requests = 0
        requests_since_last_log = 0

        def ok(res_list):
            if not res_list: return False
            r = res_list[0]
            if not getattr(r.response, "success", False): return False
            return True

        async def get_env_challenge(e: BaseEnv) -> Challenge:
            key = e.name
            chal, uses = chal_cache.get(key, (None, 0))
            if chal is None or uses >= MAX_USES:
                # Build a stable placeholder challenge; actual evaluation happens in run_episode
                chal, uses = Challenge(env=e, prompt=f"{e.name} placeholder", extra={}), 0
            chal_cache[key] = (chal, uses + 1)
            return chal

        async def schedule_env_round(e: BaseEnv):
            nonlocal total_requests, requests_since_last_log
            name = e.name
            if env_inflight[name]:
                return
            # Compute common id for this env round
            data_len = (getattr(e, "data_len", 200) or 200)
            tid = random.randint(0, int(data_len) - 1)
            chal = await get_env_challenge(e)
            tasks = {}
            for m in miners_map.values():
                if not getattr(m, "model", None):
                    continue
                t = asyncio.create_task(run([chal], m, timeout=180, task_ids={name: tid}))
                tasks[int(m.uid)] = t
                total_requests += 1
                requests_since_last_log += 1
            env_inflight[name] = tasks

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
                        try:
                            st = await ensure_subtensor()
                            blk = await st.get_current_block()
                        except BaseException as e:
                            logger.warning(f"sink_worker: get_current_block() failed, will retry later. err={e!r}")
                            traceback.print_exc()
                            continue

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
                        except Exception as e:
                            logger.warning(f"sink_worker: sink_enqueue() failed, will retry later. err={e!r}")
                            traceback.print_exc()
                            # keep going; don't drop future batches
                        batch.clear()
                        first_put_time = None
                except BaseException:
                    traceback.print_exc()
                    logger.error("sink_worker: unexpected error, continuing loop")
                    await asyncio.sleep(1)

        async def main_loop():
            global HEARTBEAT
            nonlocal last_status_log, requests_since_last_log
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

                    # Pre-warm shared container envs if enabled
                    for e in envs:
                        if isinstance(e, ContainerEnv):
                            try:
                                await e.ensure_ready()
                            except Exception as ex:
                                logger.warning(f"ensure_ready failed for {e.name}: {ex}")

                    # periodic status logging
                    if now - last_status_log >= 30:
                        elapsed = now - last_status_log if last_status_log > 0 else 30
                        rps = requests_since_last_log / elapsed
                        queue_size = sink_q.qsize()
                        inflight_total = sum(len(d) for d in env_inflight.values())
                        logger.info(f"[STATUS] miners={len(miners_map)} inflight={inflight_total} queue={queue_size} req/s={rps:.1f} total_req={total_requests}")
                        last_status_log = now
                        requests_since_last_log = 0

                    # Schedule a round per env if idle
                    for e in envs:
                        await schedule_env_round(e)

                    # Aggregate inflight tasks across envs
                    all_tasks = [t for d in env_inflight.values() for t in d.values()]
                    if not all_tasks:
                        await asyncio.sleep(0.2)
                        continue

                    done, _ = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=5.0)
                    HEARTBEAT = now = time.monotonic()  # tick during long episodes
                    for t in done:
                        # locate env + uid for task
                        found = None
                        for name, d in env_inflight.items():
                            for uid, tk in d.items():
                                if tk is t:
                                    found = (name, uid)
                                    break
                            if found: break
                        name, uid = found if found else ("?", -1)
                        # Remove inflight entry
                        if name in env_inflight:
                            env_inflight[name].pop(uid, None)
                        try:
                            res_list = await t  # list[Result]
                        except Exception as e:
                            logger.debug(f"task error env={name} uid={uid}: {e}")
                            res_list = []

                        if res_list:
                            # enqueue results for sink
                            sink_q.put_nowait(res_list)
                            # Update metrics/logs for each result
                            for r in res_list:
                                try:
                                    SCORE.labels(uid=r.miner.uid, env=r.challenge.env.name).set(r.evaluation.score)
                                except Exception:
                                    pass
                                logger.debug(
                                    LOG_TEMPLATE.format(
                                        pct=0,
                                        env=r.challenge.env.name,
                                        uid=r.miner.uid,
                                        model=(r.miner.model or "")[:50],
                                        success="RECV" if getattr(r.response, "success", False) else "NULL",
                                        score=r.evaluation.score,
                                        latency=r.response.latency_seconds,
                                    )
                                )

                        # If env round completed (no more inflight), advance id and immediately schedule next round
                        if name in env_inflight and not env_inflight[name]:
                            env_round[name] = (env_round[name] + 1) % (next((e.data_len for e in envs if e.name == name), 200) or 200)
                            e = next((e for e in envs if e.name == name), None)
                            if e is not None:
                                await schedule_env_round(e)
            except asyncio.CancelledError:
                pass
            finally:
                # cancel sink worker and wait for final flush
                sink_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sink_task

        await main_loop()

    async def main():
        timeout = int(os.getenv("AFFINE_WATCHDOG_TIMEOUT", "900"))
        await asyncio.gather(_run(), watchdog(timeout=timeout))

    asyncio.run(main())



# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #

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
                force_uid0 = 0.9
                uids, weights = await get_weights(scale=0.5, burn=force_uid0)
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
    async def _run_weights():
        if not check_env_variables():
            return

        api_key = os.getenv("CHUTES_API_KEY", "")
        is_valid = await validate_api_key(api_key)
        if not is_valid:
            logger.error("CHUTES_API_KEY validation failed. The key may be invalid or expired.")
            logger.info("Please check your CHUTES_API_KEY and ensure it has proper permissions")
            return
        
        logger.debug("All environment variables validated successfully")
        return await get_weights()

    asyncio.run(_run_weights())

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

    # Warmup via legacy SAT is removed in Quixand-only mode.
    logger.debug("Mining setup complete. Model is live!")

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
