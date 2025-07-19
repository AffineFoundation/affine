#!/usr/bin/env python3
import os
import io
import sys
import json
import time
import math
import types 
import click
import random
import aiohttp
import asyncio
import logging
import textwrap
import botocore
import numpy as np
import bittensor as bt
import botocore.config
from pathlib import Path
from rich.table import Table
from dotenv import load_dotenv
from rich.console import Console
from huggingface_hub import HfApi
from collections import defaultdict
from abc import ABC, abstractmethod
from alive_progress import alive_bar
from pydantic import BaseModel, Field
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download
from bittensor.core.errors import MetadataError
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence

NETUID = 120

# ── LOGGING ─────────────────────────────────────────────────────────────
TRACE = 5
logging.addLevelName(TRACE, "TRACE")
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("affine")
def setup_logging(verbosity: int):
    if verbosity >= 3: level = TRACE
    elif verbosity == 2: level = logging.DEBUG
    elif verbosity == 1: level = logging.INFO
    else: level = logging.CRITICAL + 1
    for noisy_logger in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
# ── SUBTENSOR ─────────────────────────────────────────────────────────────
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.debug("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor()
        await SUBTENSOR.initialize()
        logger.debug("Connected")
    return SUBTENSOR

# ── CLI ─────────────────────────────────────────────────────────────────
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)
    
# ── Config. ─────────────────────────────────────────────────────────────────
load_dotenv(os.path.expanduser("~/.affine/config.env"), override=True)
load_dotenv(override=True)
def get_conf(key, default = None) -> Any:
    value = os.getenv(key)
    if not value and default == None:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True)
        sys.exit(1)
    elif not value:
        return default
    return value

@cli.command('set')
@click.argument('key')
@click.argument('value')
def set_config(key: str, value: str):
    """Set a key-value pair in ~/.affine/config.env."""
    path = os.path.expanduser("~/.affine/config.env")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [l for l in open(path).readlines() if not l.startswith(f"{key}=")] if os.path.exists(path) else []
    lines.append(f"{key}={value}\n")
    open(path, "w").writelines(lines)
    logger.info("Set %s in %s", key, path)
    click.echo(f"Set {key} in {path}")
    
# ── Models ───────────────────────────────────────────────────────────────
def _truncate(text: Optional[str], max_length: int = 80) -> str:
    if not text:
        return ""
    return textwrap.shorten(text, width=max_length, placeholder="...")

class Miner(BaseModel):
    uid: int
    hotkey: str
    model: Optional[str] = None
    revision: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]
    success: bool
    def __repr__(self) -> str:
        resp = _truncate(self.response)
        err  = _truncate(self.error)
        return (
            f"<Response model={self.model!r} success={self.success} "
            f"latency={self.latency_seconds:.3f}s attempts={self.attempts} "
            f"response={resp!r} error={err!r}>"
        )
    __str__ = __repr__

class BaseEnv(BaseModel, ABC):
    @property
    def name(self) -> str:return self.__class__.__name__
    class Config: 
        arbitrary_types_allowed = True
    def __hash__(self): return hash(self.name)
    def __repr__(self): return self.name
    async def many(self, n: int) -> List["Challenge"]:
        return await asyncio.gather(*(self.generate() for _ in range(n)))
    @abstractmethod
    async def generate(self) -> "Challenge": ...
    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: Response) -> "Evaluation": ...

class Challenge(BaseModel):
    env: BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    def json(self, **kwargs): # Hack to get name of env.
        dd = self.dict(**kwargs); dd['env'] = self.env.name;
        return json.dumps(dd)
    async def evaluate(self, response: Response) -> "Evaluation":
        return await self.env.evaluate(self, response)
    def __repr__(self) -> str:
        pr = _truncate(self.prompt)
        return f"<Challenge env={self.env.name!r} prompt={pr!r}>"
    __str__ = __repr__

class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)
    def json(self, **kwargs): # Hack to get name of env.
        dd = self.dict(**kwargs); dd['env'] = self.env.name;
        return json.dumps(dd)
    def __repr__(self) -> str:
        ex = {k: _truncate(str(v)) for k, v in self.extra.items()}
        return f"<Evaluation env={self.env.name!r} score={self.score:.4f} extra={ex!r}>"
    __str__ = __repr__

class Result(BaseModel):
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation
    
    
# ── S3 ─────────────────────────────────────────────────────────────────
async def get_client():
    ACCOUNT    = get_conf("R2_ACCOUNT_ID")
    KEY_ID     = get_conf("R2_WRITE_ACCESS_KEY_ID")
    SECRET     = get_conf("R2_WRITE_SECRET_ACCESS_KEY")
    ENDPOINT   = f"https://{ACCOUNT}.r2.cloudflarestorage.com"
    CLIENT_CFG = botocore.config.Config(max_pool_connections=256)
    session = get_session()
    CLIENT = await session.create_client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
        config=CLIENT_CFG,
    ).__aenter__()
    return CLIENT

async def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, default=lambda o: o.json() if hasattr(o, "json") else o).encode()

async def get(key: str) -> Any:
    client = await get_client()
    BUCKET     = get_conf("R2_BUCKET_ID")
    response = await client.get_object(Bucket=BUCKET, Key=key)
    body = await response["Body"].read()
    content_type = response.get("ContentType", "")
    if content_type == "application/json": return json.loads(body.decode("utf-8"))
    else: return body

async def sink(key: str, obj: Any, *, content_type: str = "application/json"):
    try:
        client = await get_client()
        BUCKET     = get_conf("R2_BUCKET_ID")
        body = await _json_bytes(obj) if content_type == "application/json" else obj
        await client.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=content_type)
    except:
        logger.trace('R2 bucket is not set cannot sink. Continuing...')
        pass

# ── API ───────────────────────────────────────────────────────────────
async def get_chute(model: str) -> Dict[str, Any]:
    url = f"https://api.chutes.ai/chutes/{model}"
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
            logger.trace("Fetched chute info for %s", model)
            return info
        
async def get_chute_code(identifier: str) -> Optional[str]:
    """Return raw chute Python code for *identifier* or ``None``.

    The endpoint is `https://api.chutes.ai/chutes/code/{identifier}`.
    """
    url = f"https://api.chutes.ai/chutes/code/{identifier}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            return await r.text(errors="ignore")

def _extract_revision(code: str) -> Optional[str]:
    """Extract `--revision <sha>` from chute code string."""
    import re
    m = re.search(r"--revision\s+([0-9a-f]{40})", code)
    return m.group(1) if m else None

async def miners(
        uids: Optional[Union[int, List[int]]] = None, 
        no_null: bool = False, 
        netuid: int = NETUID,
        min_block: int = 0,
        metagraph: object = None,
    ) -> Dict[int, Miner]:
    subtensor = await get_subtensor()
    if metagraph is None:
        metagraph = await subtensor.metagraph(netuid)
    revs = await subtensor.get_all_revealed_commitments(netuid)
    if uids is None: uids = list(range(len(metagraph.hotkeys)))
    elif isinstance(uids, int): uids = [uids]
    async def _get_miner(uid: int) -> Optional[Miner]:
        if not (0 <= uid < len(metagraph.hotkeys)): return None
        hotkey = metagraph.hotkeys[uid]
        commits = revs.get(hotkey) or []
        if not commits: return None # Filter on null.
        block, model_data = commits[-1]
        revision = None
        # Nouveau format JSON {"model":..., "revision":...}
        if isinstance(model_data, str) and model_data.strip().startswith("{"):
            try:
                parsed = json.loads(model_data)
                model_data = parsed.get("model")
                revision = parsed.get("revision")
            except Exception:
                pass
        model = model_data
        if no_null and block is None: return None # Filter on null.
        if no_null and model is None: return None # Filter on null.
        if block < min_block: return None # Filter on block.
        chute = await get_chute(str(model))
        if chute is None:
            # fall back to code endpoint using model as identifier
            code = await get_chute_code(str(model))
            revision = _extract_revision(code or "") if code else None
        else:
            # Try extract revision from returned info first
            if 'revision' in chute and chute['revision']:
                revision = str(chute['revision'])
            else:
                # fallback: fetch code via uuid if available
                cid = chute.get('id') or chute.get('uuid')
                if cid:
                    code = await get_chute_code(cid)
                    revision = _extract_revision(code or "") if code else None
        if no_null and chute is None: return None # Filter on no chute.
        miner = Miner(uid=uid, hotkey=hotkey, model=str(model), revision=revision, block=int(block), chute=chute)
        return miner
    miners = {uid: miner for uid in uids if (miner := await _get_miner(uid)) is not None}
    return miners

# ── run challenges ────────────────────────────────────────────────────────────
HTTP_SEM = asyncio.Semaphore(int(os.getenv("AFFINE_HTTP_CONCURRENCY", "16")))
TERMINAL = {400, 404, 410}

async def run(challenges, miners, timeout=120, retries=0, backoff=1, progress=True):
    if not isinstance(challenges, list): challenges = [challenges]
    if isinstance(miners, Miner): miners = [miners]
    if isinstance(miners, dict):  mmap = miners
    elif isinstance(miners, list) and all(hasattr(m, "uid") for m in miners):  mmap = {m.uid: m for m in miners}
    else: mmap = await miners(miners)
    if logger.isEnabledFor(logging.TRACE):
        logger.trace("Running %d challenges on %d miners",
                     len(challenges), len(mmap))
    logger.trace("Running challenges: %s on miners: %s", [chal.prompt[:30] for chal in challenges], list(mmap.keys()))

    async def _run_one(chal, model):
        url = "https://llm.chutes.ai/v1/chat/completions"
        hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
        start = time.monotonic()
        logger.trace("Starting challenge on %s", model)
        for attempt in range(1, retries + 2):
            try:
                async with HTTP_SEM, sess.post(url, json={"model": model, "messages":[{"role":"user","content":chal.prompt}]}, headers=hdr, timeout=timeout) as r:
                    text = await r.text(errors="ignore")
                    logger.trace("HTTP %d on attempt %d for %s", r.status, attempt, model)
                    if r.status in TERMINAL:
                        err = f"{r.status}:{text}"
                        logger.debug("Terminal error for %s: %s", model, err)
                        return Response(response=None, latency_seconds=time.monotonic()-start, attempts=attempt, model=model, error=err, success=False)
                    if r.status != 200:
                        raise RuntimeError(f"{r.status}:{text}")
                    body = await r.json()
                    res = body["choices"][0]["message"]["content"]
                    logger.trace("Success for %s in %.2fs (attempt %d)", model, time.monotonic()-start, attempt)
                    return Response(response=res, latency_seconds=time.monotonic()-start, attempts=attempt, model=model, error=None, success=True)
            except Exception as e:
                logger.debug("Error on attempt %d for %s: %s", attempt, model, e)
                if attempt > retries:
                    return Response(response=None, latency_seconds=time.monotonic()-start, attempts=attempt, model=model, error=str(e), success=False)
                await asyncio.sleep(backoff * 2**(attempt-1) * (1 + random.uniform(-.1, .1)))

    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        async def proc(m, chal):
            resp = await _run_one(chal, m.model)
            try:
                ev = await chal.evaluate(resp)
                logger.trace("Evaluation done for %s: %r", m.model, ev)
            except Exception as e:
                logger.warning("Evaluation failed for miner %s on %s: %s", 
                              m.uid, chal.env.name, str(e))
                # Create fallback evaluation with zero score
                ev = Evaluation(
                    env=chal.env,
                    score=0.0,
                    extra={"error": str(e), "evaluation_failed": True}
                )
            return Result(miner=m, challenge=chal, response=resp, evaluation=ev)

        tasks = [asyncio.create_task(proc(m, chal))
                 for m in mmap.values() if m.model for chal in challenges]
        if progress:
            with alive_bar(len(tasks), title="Running challenges") as bar:
                for task in asyncio.as_completed(tasks):
                    results.append(await task); bar()
        else:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
    logger.trace("Finished all runs (%d results)", len(results))
    return results


# ── CLI & commands ────────────────────────────────────────────────────────────
# Import environments
from .envs.sat import SAT
from .envs.abduction import ABDUCTION
from .envs.deduction import DEDUCTION
from .envs.gpqa import GPQA

# Registry of active environments
ENVS = {"SAT": SAT, "ABDUCTION": ABDUCTION, "DEDUCTION": DEDUCTION, "GPQA": GPQA}

@cli.command("validate")
@click.option('--coldkey', default=None, help='Cold wallet name')
@click.option('--hotkey', default=None, help='Hot wallet key')
def validate(coldkey: str, hotkey: str):
    """Starts an affine validator"""
    console = Console()
    coldkey = coldkey or get_conf("BT_WALLET_COLD", "default")
    hotkey = hotkey or get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)
    async def main():
        K, alpha = 10, 0.999
        logger.debug("Validator started with K=%d, alpha=%.3f", K, alpha)
        subtensor = await get_subtensor()
        meta = await subtensor.metagraph(NETUID)
        window_start = await subtensor.get_current_block()
        miners_map = await miners(no_null=True, metagraph=meta)
        hotkeys = [m.hotkey for m in miners_map.values()]
        blocks = {m.hotkey: m.block for m in miners_map.values()}
        scores = {m.hotkey: defaultdict(float) for m in miners_map.values()}
        while True:
            try:
                logger.trace("Refreshing metagraph and miners...")
                meta = await subtensor.metagraph(NETUID)
                miners_map = await miners(no_null=True, metagraph=meta)
                for m in miners_map.values():
                    if m.hotkey not in blocks or blocks[m.hotkey] != m.block:
                        logger.debug("Miner %s block updated: %s -> %s", m.hotkey, blocks.get(m.hotkey), m.block)
                        blocks[m.hotkey] = m.block
                        scores[m.hotkey] = scores.get(m.hotkey, defaultdict(float))
                miners_map = {hk: m for hk, m in miners_map.items() if m.chute and m.chute.get('hot', False)}
                logger.debug("Active hot miners: %d", len(miners_map))

                # Deduplicate miners by model, keep lowest block
                best_by_model = {}
                for m in miners_map.values():
                    cur = best_by_model.get(m.model)
                    if cur is None or (m.block or float("inf")) < (cur.block or float("inf")):
                        best_by_model[m.model] = m
                miners_map = {m.hotkey: m for m in best_by_model.values()}
                blocks = {hk: m.block for hk, m in miners_map.items()}
                scores = {hk: scores.get(hk, defaultdict(float)) for hk in miners_map}
                hotkeys = list(miners_map)
                logger.trace("Deduplicated miners: %d", len(miners_map))

                # Collect scores for K blocks
                logger.debug("Collecting scores for window [%d, %d)", window_start, window_start + K)
                while await subtensor.get_current_block() < window_start + K:
                    blk = await subtensor.get_current_block()
                    # Build challenges for this block
                    challenges = []
                    for env_cls in ENVS.values():
                        try:
                            challenges.append(await env_cls().generate())
                        except Exception as gen_err:
                            logger.debug("Skipping env %s due to error: %s", env_cls.__name__, gen_err)

                    if not challenges:
                        logger.debug("No challenges generated at block %d, skipping", blk)
                        await asyncio.sleep(1)
                        continue

                    logger.trace("Running %d challenges at block %d", len(challenges), blk)
                    results = await run(challenges=challenges, miners=miners_map)
                    await sink(f"affine/results/{wallet.hotkey.ss58_address}/{blk:08d}.json", [r.json() for r in results])
                    logger.trace("Computed scores for window %d", window_start)
                    for r in results:
                        if r.response.success:
                            e, hk, raw = r.challenge.env.name, r.miner.hotkey, r.evaluation.score
                            scores[hk][e] = raw * (1 - alpha) + scores[hk][e] * alpha
                logger.trace("Computed scores for window %d", window_start)
                ranks, counts = {}, defaultdict(int)
                for e in ENVS:
                    uniq = sorted({scores[h][e] for h in hotkeys}, reverse=True)
                    rank_of = {score: i+1 for i, score in enumerate(uniq)}
                    ranks[e] = {h: rank_of[scores[h][e]] for h in hotkeys}
                env_list = list(ENVS)
                for a in hotkeys:
                    for b in hotkeys:
                        if a != b:
                            better = sum(ranks[e][a] < ranks[e][b] for e in env_list)
                            not_worse = all(ranks[e][a] <= ranks[e][b] for e in env_list)
                            if not_worse and better >= 1: counts[a] += 1
                best_key, best = (-1, None), None
                for h in hotkeys:
                    key = (counts.get(h, 0), -blocks.get(h, 0))
                    if key > best_key: best_key, best = key, h
                logger.trace("Computed ranks for window %d", window_start)
                weights = [1.0 if hk == best else 0.0 for hk in meta.hotkeys]
                snap_key = f"affine/snapshots/{wallet.hotkey.ss58_address}/{window_start:08d}.json"
                await sink(snap_key, {
                    "window_start": window_start,
                    "scores": {hk: dict(scores[hk]) for hk in hotkeys},
                    "blocks": blocks,
                    "weights": weights,
                    "miners": {hk: {'uid': miner.uid, 'model': miner.model} for hk, miner in miners_map.items()}
                })
                logger.trace("Snapshot saved: %s", snap_key)

                # — Render Rich table
                table = Table(title=f"Window {window_start}–{window_start+K}")
                table.add_column("Miner UID", style="bold")
                table.add_column("Block", justify="right")
                table.add_column("Model", justify="right")
                [table.add_column(f"{e} Score", justify="right") or table.add_column(f"{e} Rank", justify="right") for e in ENVS]
                table.add_column("Weight", justify="right")
                for hk in hotkeys:
                    miner = miners_map.get(hk)
                    if not miner: continue
                    row = [str(miner.uid), str(miner.block), miner.model]
                    for e in ENVS: row.extend([f"{scores[hk][e]:.4f}", str(ranks[e][hk])])
                    row.append(f"{weights[meta.hotkeys.index(hk)]:.2f}")
                    table.add_row(*row)
                console.print(table)
                await subtensor.set_weights(wallet=wallet, netuid=NETUID, uids=meta.uids, weights=weights, wait_for_inclusion=False)
                window_start += K
            except KeyboardInterrupt:
                logger.debug("Validator interrupted by user")
                sys.exit()
            except Exception as e:
                logger.debug("Transient error: %s. Continuing validator loop…", e)
                console.log(f"[yellow]Transient error:[/yellow] {e}. Continuing validator loop…")
                continue
    asyncio.run(main())
    
    
@cli.command("pull")
@click.argument("uid", type=int)
@click.option("--model_path", "-p", default = './model_path', required=True, type=click.Path(), help="Local directory to save the model")
@click.option('--hf-token', default=None, help="Hugging Face API token (env HF_TOKEN if unset)")
def pull(uid: int, model_path: str, hf_token: str):
    """Pulls a model from a specific miner UID if exists."""

    # 1. Ensure HF token
    hf_token     = hf_token or get_conf("HF_TOKEN")

    # 2. Lookup miner on‑chain
    miner_map = asyncio.run(miners(uids=uid, no_null=True))
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

@cli.command("push")
@click.option('--model_path',  default='./model_path', help='Local path to model artifacts.')
@click.option('--existing-repo', default=None, help='Use an existing HF repo instead of uploading (format <user>/<repo>)')
@click.option('--coldkey',     default=None, help='Name of the cold wallet to use.')
@click.option('--hotkey',      default=None, help='Name of the hot wallet to use.')
def push(model_path: str, existing_repo: str, coldkey: str, hotkey: str):
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
    chutes_api_key = get_conf("CHUTES_API_KEY")
    chute_user     = get_conf("CHUTE_USER")
    # TODO: validate API creds, exit gracefully if missing

    # -----------------------------------------------------------------------------
    # 2. Prepare HF repo name
    #    - If --existing-repo provided, use it directly and skip local upload
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
    info     = api.repo_info(repo_id=repo_name, repo_type="model")
    revision = getattr(info, "sha", getattr(info, "oid", "")) or ""
    logger.debug("Latest revision: %s", revision)

    # -----------------------------------------------------------------------------
    # 6. Commit model revision on-chain
    # -----------------------------------------------------------------------------
    async def commit_to_chain():
        """Submit the model commitment, retrying if the subnet space quota is full."""
        logger.debug("Preparing on-chain commitment")
        sub     = await get_subtensor()
        payload = json.dumps({"model": repo_name, "revision": revision})
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

    asyncio.run(commit_to_chain())

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
        rev_flag = f"--revision {revision} " if revision else ""
        chutes_config = textwrap.dedent(f"""
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="{chute_user}",
    readme="{repo_name}",
    model_name="{repo_name}",
    image="chutes/sglang:0.4.6.post5b",
    concurrency=20,
    node_selector=NodeSelector(
        gpu_count=4,
        min_vram_gb_per_gpu=24,
        include=["4090", "l40s", "a6000_ada"],
        exclude=["h200", "b200", "mi300x"],
    ),
    engine_args=(
        "--trust-remote-code "
        "{rev_flag}"
        "--tool-call-parser deepseekv3 "
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
        logger.trace(stdout.decode())
        if proc.returncode != 0:
            logger.debug("Chutes deploy failed with code %d", proc.returncode)
            raise RuntimeError("Chutes deploy failed")
        tmp_file.unlink(missing_ok=True)
        logger.debug("Chute deployment successful")

    asyncio.run(deploy_to_chutes())

    # -----------------------------------------------------------------------------
    # 9. Warm up model until it’s marked hot
    # -----------------------------------------------------------------------------
    async def warmup_model():
        logger.debug("Warming up model with SAT challenges")
        sub       = await get_subtensor()
        meta      = await sub.metagraph(NETUID)
        my_uid    = meta.hotkeys.index(wallet.hotkey.ss58_address)
        miner  = (await miners(uids=my_uid))[my_uid]

        while not (miner.chute or {}).get("hot", False):
            challenge = await SAT().generate()
            await run(challenges=challenge, miners=[miner])
            await sub.wait_for_block()
            miner = (await miners(uids=my_uid))[my_uid]
            logger.trace("Checked hot status: %s", (miner.chute or {}).get("hot"))

        logger.debug("Model is now hot and ready")

    asyncio.run(warmup_model())
    logger.debug("Mining setup complete. Model is live!")  


        

