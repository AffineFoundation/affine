#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import random
import click
import aiohttp
import asyncio
import bittensor as bt
from dotenv import load_dotenv
from collections import defaultdict
from abc import ABC, abstractmethod
from alive_progress import alive_bar
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from .sample_manager import SampleManager
from . import val_config

# ── load env ─────────────────────────────────────────────────────────────────
load_dotenv(os.path.expanduser("~/.affine/config.env"), override=True)
load_dotenv(override=True)

def get_conf(key) -> Any:
    value = os.getenv(key)
    if not value:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True)
        sys.exit(1)
    return value

# ── logging setup ─────────────────────────────────────────────────────────────
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
    # Suppress noisy loggers
    for noisy_logger in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "fsspec", "datasets.utils.file_utils"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ── core models ───────────────────────────────────────────────────────────────
class Miner(BaseModel):
    uid: int
    hotkey: str
    model: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]

class BaseEnv(BaseModel, ABC):
    class Config:
        arbitrary_types_allowed = True

    async def many(self, n: int) -> List["Challenge"]:
        """Generate n challenges, always using samples from JSONL"""
        return await self._many_from_samples(n)
    
    async def _many_from_samples(self, n: int) -> List["Challenge"]:
        """Get challenges from samples, generating more if needed, with background replenishment"""
        sample_manager = SampleManager()
        
        # Use the new ensure_samples_available method which handles parallel generation
        available_samples = await sample_manager.ensure_samples_available(self, n)
        
        # Create challenges from samples
        challenges = []
        sample_ids_to_remove = []
        
        for sample in available_samples:
            challenge = Challenge(
                env=self,
                prompt=sample.prompt,
                extra=sample.extra.copy()
            )
            challenges.append(challenge)
            sample_ids_to_remove.append(sample.id)
        
        # Remove used samples immediately
        if sample_ids_to_remove:
            env_name = self.__class__.__name__
            sample_manager.remove_samples(env_name, sample_ids_to_remove)
            logger.debug(f"Removed {len(sample_ids_to_remove)} used samples for {env_name}")
        
        # Get current stats
        stats = sample_manager.get_stats(self.__class__.__name__)
        
        # If samples are running low (less than certain % of target), trigger background replenishment
        if stats.total < (val_config.TARGET_STOCK * val_config.STOCK_REPLENISH_THRESHOLD):
            logger.info(f"Sample stock running low ({stats.total}/{val_config.TARGET_STOCK}), triggering background replenishment")
            # Store the background task in the environment for later cleanup
            if not hasattr(self, '_background_tasks'):
                self._background_tasks = []
            replenish_task = asyncio.create_task(sample_manager.replenish_stock_background(self))
            self._background_tasks.append(replenish_task)
        
        return challenges

    @abstractmethod
    async def generate(self) -> "Challenge":
        ...

    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: Response) -> "Evaluation":
        ...

    async def generate_input_with_llm(self, prompt: str, model: str = "unsloth/gemma-3-27b-it", timeout: float = val_config.LLM_RESPONSE_TIMEOUT) -> str:
        """Generate input using LLM call"""
        url = "https://llm.chutes.ai/v1/chat/completions"
        token = get_conf("CHUTES_API_KEY")
        hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        logger.debug(f"Making LLM API call to {url} with model {model}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                    headers=hdr,
                    timeout=timeout
                ) as r:
                    text = await r.text(errors="ignore")
                    if r.status != 200:
                        logger.error(f"LLM API error: {r.status} - {text}")
                        raise RuntimeError(f"{r.status}:{text}")
                    data = await r.json()
                    response = data["choices"][0]["message"]["content"]
                    logger.debug(f"Got LLM response of length: {len(response)} characters")
                    return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    async def generate_inputs_parallel(self, prompts: List[str], max_concurrent: int = val_config.MAX_CONCURRENT_REQUESTS, model: str = "unsloth/gemma-3-27b-it", timeout: float = val_config.LLM_RESPONSE_TIMEOUT) -> List[str]:
        """Generate multiple inputs in parallel using LLM calls"""
        logger.debug(f"Starting parallel generation of {len(prompts)} prompts with max_concurrent={max_concurrent}")
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                try:
                    logger.debug("Acquiring semaphore for LLM call")
                    result = await self.generate_input_with_llm(prompt, model, timeout)
                    logger.debug("Successfully generated input with LLM")
                    return result
                except Exception as e:
                    logger.error(f"Failed to generate input: {e}")
                    return None
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        logger.debug(f"Created {len(tasks)} tasks for parallel execution")
        results = await asyncio.gather(*tasks)
        logger.debug(f"Completed parallel generation with {len([r for r in results if r is not None])} successful results")
        return results

class Challenge(BaseModel):
    env: BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)

    async def evaluate(self, response: Response) -> "Evaluation":
        return await self.env.evaluate(self, response)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert challenge to dictionary format for sample storage"""
        data = {
            'prompt': self.prompt  # Always include the prompt
        }
        # Flatten extra fields into top level
        if self.extra:
            # Don't overwrite prompt if it exists in extra
            extra_data = {k: v for k, v in self.extra.items() if k != 'prompt'}
            data.update(extra_data)
        return data

class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)

class Result(BaseModel):
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation

# ── fetch chute & miners ─────────────────────────────────────────────────────
async def get_chute(model: str) -> Dict[str, Any]:
    url = f"https://api.chutes.ai/chutes/{model}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            text = await r.text(errors="ignore")
            if r.status != 200:
                raise RuntimeError(f"{r.status}:{text}")
            info = await r.json()
            for k in ('readme','cords','tagline','instances'):
                info.pop(k, None)
            info.get('image', {}).pop('readme', None)
            logger.trace("Fetched chute info for %s", model)
            return info

async def miners(uids: Optional[Union[int, List[int]]] = None, no_null: bool = False) -> Dict[int, Miner]:
    NETUID = val_config.BITTENSOR_NETUID
    logger.debug(f"Initializing Bittensor subtensor for NETUID {NETUID}...")
    s = bt.async_subtensor()
    logger.debug("Connecting to Bittensor network...")
    await s.initialize()
    logger.debug("Fetching metagraph...")
    meta = await s.metagraph(NETUID)
    logger.debug("Fetching revealed commitments...")
    revs = await s.get_all_revealed_commitments(NETUID)
    logger.debug("Bittensor network calls completed")

    if uids is None:
        uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids = [uids]

    out: Dict[int, Miner] = {}
    for uid in uids:
        hk = meta.hotkeys[uid] if 0 <= uid < len(meta.hotkeys) else ""
        commits = revs.get(hk) or []
        blk, mdl = commits[-1] if commits else (None, None)
        # mdl = "unsloth/gemma-3-27b-it"
        logger.debug("Revealed model: %s", mdl)
        if no_null and blk is None:
            continue
        out[uid] = Miner(uid=uid, hotkey=hk,
                         model=str(mdl) if mdl is not None else None,
                         block=int(blk) if blk is not None else None)

    with_models = [m for m in out.values() if m.model]
    if with_models:
        logger.debug(f"Fetching chute info for {len(with_models)} miners with models...")
        infos = await asyncio.gather(
            *[get_chute(m.model) for m in with_models],
            return_exceptions=True
        )
        logger.debug("Chute info fetching completed")
        for m, info in zip(with_models, infos):
            if not isinstance(info, Exception):
                m.chute = info

    miner_info = [
        f"\tUID: {m.uid}, Hotkey: {m.hotkey}, Model: {m.model}, Block: {m.block}"
        for m in out.values()
    ]
    logger.debug("Discovered %d miners:\n%s", len(out), "\n".join(miner_info))
    return out

# ── run challenges ────────────────────────────────────────────────────────────
async def _run_one(
    session: aiohttp.ClientSession,
    chal: Challenge,
    model: str,
    timeout: float,
    retries: int,
    backoff: float
) -> Response:
    url = "https://llm.chutes.ai/v1/chat/completions"
    token = get_conf("CHUTES_API_KEY")
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    start = time.monotonic()
    for attempt in range(1, retries + 2):
        try:
            async with session.post(
                url,
                json={"model": model, "messages": [{"role": "user", "content": chal.prompt}]},
                headers=hdr,
                timeout=timeout
            ) as r:
                text = await r.text(errors="ignore")
                if r.status != 200:
                    raise RuntimeError(f"{r.status}:{text}")
                data = await r.json()
                res = data["choices"][0]["message"]["content"]
                latency = time.monotonic() - start
                logger.trace("Model %s answered in %.2fs on attempt %d", model, latency, attempt)
                return Response(response=res, latency_seconds=latency, attempts=attempt, model=model, error=None)
        except Exception as e:
            logger.debug("Attempt %d for %s failed: %s", attempt, model, e)
            if attempt > retries:
                latency = time.monotonic() - start
                return Response(response=None, latency_seconds=latency, attempts=attempt, model=model, error=str(e))
            delay = backoff * (2 ** (attempt - 1)) * (1 + random.uniform(-0.1, 0.1))
            await asyncio.sleep(delay)
    return Response(response=None, latency_seconds=time.monotonic()-start, attempts=retries+1, model=model, error="unreachable")

async def run(
    challenges: Union[Challenge, List[Challenge]],
    miners: Optional[Union[Dict[int, Miner], List[Miner], int, List[int]]] = None,
    timeout: float = val_config.LLM_RESPONSE_TIMEOUT,
    retries: int = 3,
    backoff: float = 1.0,
    progress: bool = True,
) -> List[Result]:
    if not isinstance(challenges, list):
        challenges = [challenges]
    if isinstance(miners, dict):
        mmap = miners
    elif isinstance(miners, list) and all(isinstance(m, Miner) for m in miners):
        mmap = {m.uid: m for m in miners}
    else:
        mmap = await miners(miners)
    valid = [m for m in mmap.values() if m.model]
    results: List[Result] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as sess:
        async def run_one(m, c):
            resp = await _run_one(sess, c, m.model, timeout, retries, backoff)
            ev = await c.evaluate(resp)
            return Result(miner=m, challenge=c, response=resp, evaluation=ev)
        tasks = [asyncio.create_task(run_one(m, c)) for m in valid for c in challenges]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
    logger.info("Finished %d runs", len(results))
    return results

# ── CLI & commands ────────────────────────────────────────────────────────────
from .envs.coin import COIN
from .envs.sat import SAT
from .envs.abd import ABD

ENVS = {"COIN": COIN, "SAT": SAT, "ABD": ABD}

# ── persistent Elo helpers ──────────────────────────────────────────
ELO_FILE = os.path.expanduser("~/.affine/results/elo.json")

def load_elo() -> Dict[str, float]:
    """Load Elo table from disk, creating it if absent."""
    if not os.path.exists(ELO_FILE):
        # Ensure directory exists and create an empty file
        os.makedirs(os.path.dirname(ELO_FILE), exist_ok=True)
        with open(ELO_FILE, "w") as f:
            json.dump({}, f)
        return defaultdict(lambda: 1500.0)

    try:
        with open(ELO_FILE) as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        # Corrupted file: start fresh
        return defaultdict(lambda: 1500.0)

def save_elo(table: Dict[str, float]) -> None:
    """Persist Elo dict to disk."""
    os.makedirs(os.path.dirname(ELO_FILE), exist_ok=True)
    # convert defaultdict to regular dict for json
    serial = dict(table)
    with open(ELO_FILE, "w") as f:
        json.dump(serial, f, indent=2)

@click.group()
@click.option('-v', '--verbose', count=True,
              help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """Affine CLI"""
    setup_logging(verbose)

@cli.command('run')
@click.argument('uids', callback=lambda _ctx, _param, val: [int(x) for x in val.split(',')])
@click.argument('env', type=click.Choice(list(ENVS), case_sensitive=False))
@click.option('-n', '--num', 'n', default=1, show_default=True, help='Number of challenges')
def run_command(uids, env, n):
    """Run N challenges in ENV against miners."""
    # Get recommended timeout based on environment
    env_class = ENVS[env.upper()]
    timeout = getattr(env_class, 'LLM_RESPONSE_TIMEOUT', 600.0)
    logger.debug("Using recommended timeout: %ds for %s environment", timeout, env)
    
    logger.debug("Running %d challenges in %s for UIDs %s with timeout %ds", n, env, uids, timeout)
    
    async def _coro():
        env_instance = ENVS[env.upper()]()
        chals = await env_instance.many(n)
        ms = await miners(uids)
        results = await run(challenges=chals, miners=ms, timeout=timeout)
        
        # Wait for any background replenishment tasks to complete
        if hasattr(env_instance, '_background_tasks'):
            background_tasks = [task for task in env_instance._background_tasks if not task.done()]
            if background_tasks:
                logger.info(f"Waiting for {len(background_tasks)} background tasks to complete...")
                try:
                    await asyncio.gather(*background_tasks, return_exceptions=True)
                    logger.info("Background tasks completed")
                except Exception as e:
                    logger.warning(f"Some background tasks failed: {e}")
        
        return results
    results = asyncio.run(_coro())
    print_results(results)
    save(results)

@cli.command('deploy')
@click.argument('filename', type=click.Path(exists=True))
@click.option('--chutes-api-key', default=None, help='Chutes API key')
@click.option('--hf-user', default=None, help='HuggingFace user')
@click.option('--hf-token', default=None, help='HuggingFace token')
@click.option('--chute-user', default=None, help='Chutes user')
@click.option('--wallet-cold', default=None, help='Bittensor coldkey')
@click.option('--wallet-hot', default=None, help='Bittensor hotkey')
@click.option('--existing-repo', default=None, help='Use existing HuggingFace repository')
@click.option('--blocks-until-reveal', default=720, help='Blocks until reveal on Bittensor')
def deploy(filename, chutes_api_key, hf_user, hf_token, chute_user, wallet_cold, wallet_hot, existing_repo, blocks_until_reveal):
    """Deploy a model or file using provided credentials."""
    logger.debug("Deploying %s", filename)
    
    # Resolve configuration values
    chutes_api_key = chutes_api_key or get_conf("CHUTES_API_KEY")
    hf_user        = hf_user        or get_conf("HF_USER")
    hf_token       = hf_token       or get_conf("HF_TOKEN")
    chute_user     = chute_user     or get_conf("CHUTES_USER")
    wallet_cold    = wallet_cold    or get_conf("BT_COLDKEY")
    wallet_hot     = wallet_hot     or get_conf("BT_HOTKEY")
    
    # Import deployment functions
    from .deployment import deploy_model, DeploymentConfig
    
    # Create deployment configuration
    config = DeploymentConfig(
        chutes_api_key=chutes_api_key,
        hf_user=hf_user,
        hf_token=hf_token,
        chute_user=chute_user,
        wallet_cold=wallet_cold,
        wallet_hot=wallet_hot
    )
    
    # Run deployment
    async def _deploy():
        repo_id = await deploy_model(
            local_path=filename,
            config=config,
            existing_repo=existing_repo,
            blocks_until_reveal=blocks_until_reveal
        )
        click.echo(f"Deployment completed successfully!")
        click.echo(f"Repository: {repo_id}")
        return repo_id
    
    try:
        asyncio.run(_deploy())
    except KeyboardInterrupt:
        click.echo("\n Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Deployment failed: {str(e)}", err=True)
        logger.error("Deployment failed", exc_info=True)
        sys.exit(1)

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

@cli.command('gen')
@click.argument('env', type=click.Choice(list(ENVS), case_sensitive=False), required=False)
@click.option('-n', '--num', type=int, default=None, help='Number of samples to generate per environment')
@click.option('--stock', default=val_config.TARGET_STOCK, show_default=True, help='Target stock level (generate only if below this)')
@click.option('--max-concurrent', default=val_config.MAX_CONCURRENT_REQUESTS, show_default=True, help='Maximum concurrent sample generation tasks')
def gen_samples(env, num, stock, max_concurrent):
    """Generate samples for environments. If ENV not specified, generates for all.
    If -n is specified, generates exactly that many samples regardless of current stock.
    If -n is not specified, generates samples to meet the target stock level."""
    sample_manager = SampleManager()
    
    # Determine which environments to process
    if env:
        env_names = [env.upper()]
    else:
        env_names = list(ENVS.keys())
    
    async def _generate():
        for env_name in env_names:
            logger.info(f"Processing {env_name}")
            
            # Check current stock
            stats = sample_manager.get_stats(env_name)
            
            if num is not None:
                # If -n is specified, generate exactly that many samples
                actual_generate = num
                click.echo(f"{env_name}: Generating {actual_generate} samples (current: {stats.total})")
            else:
                # If -n is not specified, generate to meet target stock
                if stats.total >= stock:
                    click.echo(f"✓ {env_name}: {stats.total} available samples (target: {stock}) - skipping")
                    continue
                
                needed = stock - stats.total
                actual_generate = needed
                click.echo(f"{env_name}: Generating {actual_generate} samples to meet target (current: {stats.total}, target: {stock})")
            
            # Generate samples using the new parallel method
            env_instance = ENVS[env_name]()
            samples = await sample_manager.generate_samples_for_env(env_instance, actual_generate, max_concurrent)
            
            # Store the samples
            sample_manager.add_samples(env_name, samples)
            
            # Show updated stats
            new_stats = sample_manager.get_stats(env_name)
            click.echo(f"{env_name}: Generated {len(samples)} samples. Total available: {new_stats.total}")
    
    asyncio.run(_generate())

@cli.command('samples')
@click.argument('env', type=click.Choice(list(ENVS), case_sensitive=False), required=False)
def samples_info(env):
    """Show sample statistics. If ENV not specified, shows all environments."""
    sample_manager = SampleManager()
    
    # Show statistics
    if env:
        stats = {env.upper(): sample_manager.get_stats(env.upper())}
    else:
        stats = sample_manager.get_all_stats()
    
    if not stats:
        click.echo("No sample data found. Run 'af gen' to generate samples.")
        return
    
    click.echo("Sample Statistics:")
    click.echo("-" * 50)
    header = f"{'Environment':<12} {'Available':<10} {'Target':<8} {'Status':<12}"
    click.echo(header)
    click.echo("-" * 50)
    
    for env_name, stat in stats.items():
        if stat.total == 0:
            status = "No samples"
        elif stat.total >= stat.target_stock:
            status = "Good"
        elif stat.total >= stat.target_stock * 0.5:
            status = "Low"
        else:
            status = "Critical"
        
        click.echo(f"{env_name:<12} {stat.total:<10} {stat.target_stock:<8} {status:<12}")

@cli.command('validate')
@click.option("--k-factor", "-k", default=val_config.ELO_K_FACTOR, show_default=True, help="Elo K-factor")
@click.option("--samples", "-s", default=val_config.DEFAULT_SAMPLES, show_default=True, help="Number of samples each miner will face")
@click.option("--delay", "-d", default=val_config.RETRY_DELAY, show_default=True, help="Delay between validation cycles (s)")
@click.option("--uids", "-u", default=None, help="Comma-separated UIDs to validate (e.g., '1,2,3,4,5'). If not specified, uses all available miners.")
@click.option("--env", "-e", type=click.Choice(list(ENVS), case_sensitive=False), default="SAT", show_default=True, help="Environment to use for validation")
@click.option("--cycles", "-c", default=1, show_default=True, type=int, help="Number of validation cycles to run. Use 0 for continuous mode.")
def validator(k_factor: int, samples: int, delay: float, uids: str, env: str, cycles: int):
    """Validate miners by giving each the same number of samples and comparing performance."""
    # Parse UIDs if provided
    target_uids = None
    if uids:
        try:
            target_uids = [int(x.strip()) for x in uids.split(',')]
            logger.info(f"Validating specific UIDs: {target_uids}")
        except ValueError:
            click.echo(f"Invalid UIDs format: {uids}. Use comma-separated integers like '1,2,3,4,5'", err=True)
            sys.exit(1)
    else:
        logger.info("Validating all available miners")
    
    # Get environment instance
    env_instance = ENVS[env.upper()]()
    logger.info(f"Using environment: {env.upper()}")
    
    elo: Dict[str, float] = load_elo()
    if isinstance(elo, dict) and not isinstance(elo, defaultdict):
        elo = defaultdict(lambda: 1500.0, elo)

    def update_elo_pairwise(a: Miner, b: Miner, score_a: float, score_b: float):
        """Update Elo ratings for two miners based on their scores"""
        ra, rb = elo[a.hotkey], elo[b.hotkey]
        qa, qb = 10**(ra/400), 10**(rb/400)
        ea, eb = qa/(qa+qb), 1 - qa/(qa+qb)
        
        # Determine winner (1.0 for win, 0.5 for tie, 0.0 for loss)
        if score_a > score_b:
            result_a = 1.0
        elif score_a < score_b:
            result_a = 0.0
        else:
            # Tie - use tiebreaker
            if a.block is not None and b.block is not None:
                result_a = 1.0 if a.block < b.block else 0.0
            else:
                result_a = 1.0 if a.uid < b.uid else 0.0
        
        elo[a.hotkey] = ra + k_factor * (result_a - ea)
        elo[b.hotkey] = rb + k_factor * ((1 - result_a) - eb)
        save_elo(elo)
        logger.trace("ELO updated: %s→%.1f, %s→%.1f", a.hotkey, elo[a.hotkey], b.hotkey, elo[b.hotkey])

    async def run_validation_cycle(available_miners: List[Miner]) -> Dict[int, float]:
        """Run one validation cycle: give each miner the same samples and return their scores."""
        click.echo(f"🔄 Running validation cycle with {len(available_miners)} miners ({samples} samples each)")
        
        # Generate the same set of challenges for all miners
        challenges = await env_instance.many(samples)
        click.echo(f"📝 Generated {len(challenges)} challenges")
        
        # Run all miners on the same challenges
        results = await run(challenges=challenges, miners=available_miners, progress=False)
        
        # Aggregate scores by miner
        miner_scores = {}
        for result in results:
            uid = result.miner.uid
            if uid not in miner_scores:
                miner_scores[uid] = 0.0
            miner_scores[uid] += result.evaluation.score
        
        # Show results
        click.echo(f"📊 Cycle Results:")
        sorted_results = sorted(miner_scores.items(), key=lambda x: x[1], reverse=True)
        for uid, score in sorted_results:
            click.echo(f"   UID {uid}: {score:.3f}")
        
        return miner_scores

    async def main_loop():
        """Run validation cycles with the new simplified approach.

        Each cycle:
          1. Fetch miners with models
          2. Give each miner the same set of samples
          3. Compare all scores pairwise and update Elo ratings
        """
        cycle_count = 0
        while True:
            # Fetch miners - either specific UIDs or all miners
            if target_uids:
                click.echo(f"🔍 Fetching miner info for UIDs: {target_uids}...")
                logger.info(f"Fetching miners for UIDs: {target_uids}")
                all_miners_dict = await miners(target_uids, no_null=False)
                logger.info(f"Fetched {len(all_miners_dict)} miners")
            else:
                click.echo("🔍 Fetching all available miners...")
                logger.info("Fetching all miners")
                all_miners_dict = await miners(no_null=False)
                logger.info(f"Fetched {len(all_miners_dict)} miners")

            # Keep only miners that have registered a model
            available = [m for m in all_miners_dict.values() if m.model]
            click.echo(f"📋 Found {len(available)} miners with models")
            logger.info(f"Found {len(available)} miners with models")
            
            # Additional filtering by target UIDs if some miners didn't have models
            if target_uids:
                available = [m for m in available if m.uid in target_uids]
                click.echo(f"🎯 Filtered to {len(available)} target miners")
                logger.info(f"Filtered to {len(available)} target miners")
                
            if len(available) < 2:
                click.echo(f"⚠️  Only {len(available)} miner(s) available with models. Need at least 2 for validation.")
                logger.debug("Only %d miner(s) with model; retrying in %ds", len(available), delay)
                await asyncio.sleep(delay)
                continue

            # Limit to max batch size
            if len(available) > val_config.MAX_MINERS_PER_BATCH:
                available = available[:val_config.MAX_MINERS_PER_BATCH]
                click.echo(f"🔢 Limited to {len(available)} miners (max batch size)")

            # Run validation cycle
            miner_scores = await run_validation_cycle(available)
            
            # Update Elo ratings by comparing all pairs
            uid_to_miner = {m.uid: m for m in available}
            comparisons_made = 0
            
            click.echo(f"⚡ Updating Elo ratings...")
            for i, (uid_a, score_a) in enumerate(miner_scores.items()):
                for uid_b, score_b in list(miner_scores.items())[i+1:]:
                    miner_a = uid_to_miner[uid_a]
                    miner_b = uid_to_miner[uid_b] 
                    update_elo_pairwise(miner_a, miner_b, score_a, score_b)
                    comparisons_made += 1
            
            click.echo(f"📈 Made {comparisons_made} pairwise Elo comparisons")

            # Show leaderboard
            if elo:
                # Filter leaderboard to only show relevant miners if UIDs were specified
                if target_uids:
                    # Get hotkeys for target UIDs
                    target_hotkeys = {m.hotkey for m in available if m.uid in target_uids}
                    filtered_elo = {hk: rating for hk, rating in elo.items() if hk in target_hotkeys}
                    sorted_elo = sorted(filtered_elo.items(), key=lambda kv: kv[1], reverse=True)
                    click.echo(f"\n🏅 Elo leaderboard (UIDs: {','.join(map(str, target_uids))}, {env.upper()}):")
                else:
                    sorted_elo = sorted(elo.items(), key=lambda kv: kv[1], reverse=True)
                    click.echo(f"\n🏅 Elo leaderboard ({env.upper()}):")
                
                click.echo("{:<5} {:<5} {:<40} {:>8}".format("#", "UID", "Hotkey", "Elo"))
                click.echo("-"*60)
                hotkey_to_uid = {m.hotkey: m.uid for m in available}
                for rank, (hk, rating) in enumerate(sorted_elo, 1):
                    uid = hotkey_to_uid.get(hk, "?")
                    display_hotkey = hk[:40] if len(hk) > 40 else hk
                    click.echo(f"{rank:<5} {uid:<5} {display_hotkey:<40} {rating:>8.1f}")

            # Increment cycle counter
            cycle_count += 1
            
            # Check if we've completed the specified number of cycles
            if cycles > 0 and cycle_count >= cycles:
                click.echo(f"\n✅ Completed {cycle_count} validation cycles.")
                break
            
            # Wait before the next cycle
            if cycles == 0 or cycle_count < cycles:
                click.echo(f"⏳ Waiting {delay}s before next cycle...")
                await asyncio.sleep(delay)

    if target_uids:
        uids_str = ','.join(map(str, target_uids))
        if cycles > 0:
            logger.info("Starting validator (k=%d, s=%d, d=%.1f, env=%s, uids=%s, cycles=%d)", k_factor, samples, delay, env.upper(), uids_str, cycles)
        else:
            logger.info("Starting validator (k=%d, s=%d, d=%.1f, env=%s, uids=%s, cycles=∞)", k_factor, samples, delay, env.upper(), uids_str)
    else:
        if cycles > 0:
            logger.info("Starting validator (k=%d, s=%d, d=%.1f, env=%s, uids=all, cycles=%d)", k_factor, samples, delay, env.upper(), cycles)
        else:
            logger.info("Starting validator (k=%d, s=%d, d=%.1f, env=%s, uids=all, cycles=∞)", k_factor, samples, delay, env.upper())
    
    asyncio.run(main_loop())
    
    # Show final summary if limited cycles were specified
    if cycles > 0:
        click.echo("\n" + "="*60)
        click.echo("🎯 FINAL VALIDATION SUMMARY")
        click.echo("="*60)
        
        # Show final Elo ratings
        final_elo = load_elo()
        if final_elo:
            if target_uids:
                # Get miner info for target UIDs
                async def get_final_miners():
                    return await miners(target_uids, no_null=False)
                final_miners = asyncio.run(get_final_miners())
                target_hotkeys = {m.hotkey for m in final_miners.values() if m.uid in target_uids}
                filtered_elo = {hk: rating for hk, rating in final_elo.items() if hk in target_hotkeys}
                sorted_final_elo = sorted(filtered_elo.items(), key=lambda kv: kv[1], reverse=True)
                hotkey_to_uid = {m.hotkey: m.uid for m in final_miners.values()}
                
                click.echo(f"Final Elo Rankings (UIDs: {','.join(map(str, target_uids))}, {env.upper()}):")
            else:
                sorted_final_elo = sorted(final_elo.items(), key=lambda kv: kv[1], reverse=True)
                # Get all miners for UID mapping
                async def get_all_miners():
                    return await miners(no_null=False)
                all_miners = asyncio.run(get_all_miners())
                hotkey_to_uid = {m.hotkey: m.uid for m in all_miners.values()}
                
                click.echo(f"Final Elo Rankings ({env.upper()}):")
            
            click.echo("{:<5} {:<5} {:<35} {:>8}".format("#", "UID", "Hotkey", "Elo"))
            click.echo("-" * 55)
            
            for rank, (hotkey, rating) in enumerate(sorted_final_elo, 1):
                uid = hotkey_to_uid.get(hotkey, "?")
                display_hotkey = hotkey[:35] if len(hotkey) > 35 else hotkey
                click.echo(f"{rank:<5} {uid:<5} {display_hotkey:<35} {rating:>8.1f}")
        
        click.echo(f"\n✅ Validation completed: {cycles} cycles, {env.upper()} environment")
        if target_uids:
            click.echo(f"   Miners tested: {','.join(map(str, target_uids))}")
        click.echo(f"   Samples per miner: {samples}")
        click.echo(f"   K-factor: {k_factor}")
        click.echo("\n💾 Elo ratings saved to ~/.affine/results/elo.json")

@cli.command('elo')
@click.option("--uids", "-u", default=None, help="Comma-separated UIDs to show (e.g., '1,2,3,4,5'). If not specified, shows all.")
def show_elo(uids: str):
    """Show current Elo ratings."""
    elo = load_elo()
    
    if not elo:
        click.echo("No Elo ratings found. Run 'af validate' to generate Elo scores.")
        return
    
    # Parse UIDs if provided
    target_uids = None
    if uids:
        try:
            target_uids = [int(x.strip()) for x in uids.split(',')]
        except ValueError:
            click.echo(f"Invalid UIDs format: {uids}. Use comma-separated integers like '1,2,3,4,5'", err=True)
            sys.exit(1)
    
    # Get miner information to map hotkeys to UIDs
    async def get_miner_info():
        if target_uids:
            return await miners(target_uids, no_null=False)
        else:
            return await miners(no_null=False)
    
    miners_dict = asyncio.run(get_miner_info())
    hotkey_to_uid = {m.hotkey: m.uid for m in miners_dict.values()}
    
    # Filter Elo ratings if UIDs specified
    if target_uids:
        target_hotkeys = {m.hotkey for m in miners_dict.values() if m.uid in target_uids}
        filtered_elo = {hk: rating for hk, rating in elo.items() if hk in target_hotkeys}
        sorted_elo = sorted(filtered_elo.items(), key=lambda kv: kv[1], reverse=True)
        if target_uids:
            click.echo(f"🏅 Elo Ratings (UIDs: {','.join(map(str, target_uids))}):")
        else:
            click.echo("🏅 Elo Ratings (Filtered):")
    else:
        sorted_elo = sorted(elo.items(), key=lambda kv: kv[1], reverse=True)
        click.echo("🏅 Elo Ratings (All):")
    
    click.echo("{:<5} {:<5} {:<40} {:>8}".format("#", "UID", "Hotkey", "Elo"))
    click.echo("-" * 60)
    
    for rank, (hotkey, rating) in enumerate(sorted_elo, 1):
        uid = hotkey_to_uid.get(hotkey, "?")
        # Truncate hotkey for display
        display_hotkey = hotkey[:40] if len(hotkey) > 40 else hotkey
        click.echo(f"{rank:<5} {uid:<5} {display_hotkey:<40} {rating:>8.1f}")
    
    click.echo(f"\nTotal miners with Elo ratings: {len(sorted_elo)}")

@cli.command('test-miners')
@click.option("--uids", "-u", default=None, help="Comma-separated UIDs to test (e.g., '1,2,3,4,5'). If not specified, tests all.")
def test_miners(uids: str):
    """Test miners connection and display basic info (debugging tool)."""
    # Parse UIDs if provided
    target_uids = None
    if uids:
        try:
            target_uids = [int(x.strip()) for x in uids.split(',')]
            click.echo(f"🔍 Testing specific UIDs: {target_uids}")
        except ValueError:
            click.echo(f"Invalid UIDs format: {uids}. Use comma-separated integers like '1,2,3,4,5'", err=True)
            sys.exit(1)
    else:
        click.echo("🔍 Testing all available miners")
    
    async def test_miners_async():
        try:
            click.echo("📡 Connecting to Bittensor network...")
            
            if target_uids:
                miners_dict = await miners(target_uids, no_null=False)
            else:
                miners_dict = await miners(no_null=False)
            
            click.echo(f"✅ Successfully fetched {len(miners_dict)} miners")
            
            # Show basic info
            click.echo("\n📋 Miner Information:")
            click.echo("{:<5} {:<20} {:<10}".format("UID", "Model", "Block"))
            click.echo("-" * 40)
            
            for uid, miner in sorted(miners_dict.items()):
                model = miner.model[:18] + "..." if miner.model and len(miner.model) > 20 else (miner.model or "None")
                click.echo(f"{uid:<5} {model:<20} {miner.block or 'N/A':<10}")
            
            # Count miners with models
            with_models = [m for m in miners_dict.values() if m.model]
            click.echo(f"\n🎯 Found {len(with_models)} miners with models")
            
            if len(with_models) >= 2:
                click.echo("✅ Sufficient miners for validation")
            else:
                click.echo("⚠️  Need at least 2 miners with models for validation")
                
        except Exception as e:
            click.echo(f"❌ Error: {str(e)}")
            logger.error(f"Test miners failed: {str(e)}", exc_info=True)
    
    asyncio.run(test_miners_async())

# ── helpers ─────────────────────────────────────────────────────────────────
def print_results(results: List[Result]):
    stats = defaultdict(lambda: {'model': None, 'scores': []})
    for r in results:
        stats[r.miner.uid]['model'] = r.miner.model
        stats[r.miner.uid]['scores'].append(r.evaluation.score)

    fmt = "{:<5} {:<25} {:<5} {:>7} {:>8}"
    header = fmt.format("UID", "Model", "#", "Total", "Average")
    click.echo(header)
    click.echo("-" * len(header))
    for uid, data in stats.items():
        cnt   = len(data['scores'])
        total = sum(data['scores'])
        avg   = total / cnt if cnt else 0
        click.echo(fmt.format(uid, data['model'], cnt, f"{total:.4f}", f"{avg:.4f}"))

def save(results: List[Result]):
    path = os.path.expanduser("~/.affine/results")
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f"{int(time.time())}.json")
    serial = [r.model_dump() for r in results]
    for sr, r in zip(serial, results):
        sr['challenge']['env'] = r.challenge.env.__class__.__name__
    with open(file, "w") as f:
        json.dump(serial, f, indent=2)
    logger.info("Results saved to %s", file)
    click.echo(f"\nResults saved to: {file}\n")

if __name__ == "__main__":
    cli()
