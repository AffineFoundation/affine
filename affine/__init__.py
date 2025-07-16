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
from .envs.gpqa import GPQA

ENVS = {"COIN": COIN, "SAT": SAT, "ABD": ABD, "GPQA": GPQA}

# ── GRPO evaluation system ─────────────────────────────────────────
from .evaluator import Evaluator

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
@click.option("--samples", "-s", default=val_config.DEFAULT_SAMPLES, show_default=True, help="Number of samples each miner will face per environment")
@click.option("--delay", "-d", default=val_config.RETRY_DELAY, show_default=True, help="Delay between validation cycles (s)")
@click.option("--uids", "-u", default=None, help="Comma-separated UIDs to validate (e.g., '1,2,3,4,5'). If not specified, uses all available miners.")
@click.option("--env", "-e", default="ALL", show_default=True, help="Environment to test (SAT/ABD/COIN/ALL), or comma-separated for multiple (e.g., 'SAT,ABD')")
@click.option("--cycles", "-c", default=1, show_default=True, type=int, help="Number of validation cycles to run. Use 0 for continuous mode.")
@click.option("--ema-alpha", default=val_config.EVALUATOR_EMA_ALPHA, show_default=True, help="Exponential moving average smoothing factor")
@click.option("--skew-penalty", default=val_config.EVALUATOR_SKEW_PENALTY_WEIGHT, show_default=True, help="Weight for domain skew penalty")
def validator(samples: int, delay: float, uids: str, env: str, cycles: int, ema_alpha: float, skew_penalty: float):
    """Validate miners using GRPO (Group Relative Performance Optimization) across multiple environments."""
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
    
    # Parse environments
    if env.upper() == "ALL":
        env_names = list(ENVS.keys())  # Use all available environments
    else:
        env_names = [e.strip().upper() for e in env.split(',')]
        for environment in env_names:
            if environment not in ENVS:
                click.echo(f"Invalid environment: {environment}. Available environments: {list(ENVS.keys())}", err=True)
                sys.exit(1)
    
    logger.info(f"Using environments: {env_names}")
    
    # Initialize GRPO evaluator
    evaluator = Evaluator(
        required_domains=env_names,
        ema_alpha=ema_alpha,
        skew_penalty_weight=skew_penalty
    )

    async def run_validation_cycle(available_miners: List[Miner]) -> List[Any]:
        """Run one validation cycle: test each miner across all environments and return all results."""
        click.echo(f"🔄 Running GRPO validation cycle with {len(available_miners)} miners ({samples} samples per environment)")
        
        all_results = []
        
        # Test each environment
        for env_name in env_names:
            click.echo(f"🧪 Testing environment: {env_name}")
            env_instance = ENVS[env_name]()
            
            # Generate challenges for this environment
            challenges = await env_instance.many(samples)
            click.echo(f"📝 Generated {len(challenges)} {env_name} challenges")
            
            # Run all miners on these challenges
            env_results = await run(challenges=challenges, miners=available_miners, progress=False)
            all_results.extend(env_results)
            
            # Show environment results
            env_scores = {}
            for result in env_results:
                uid = result.miner.uid
                if uid not in env_scores:
                    env_scores[uid] = 0.0
                env_scores[uid] += result.evaluation.score
            
            click.echo(f"📊 {env_name} Results:")
            sorted_results = sorted(env_scores.items(), key=lambda x: x[1], reverse=True)
            for uid, score in sorted_results:
                click.echo(f"   UID {uid}: {score:.3f}")
        
        return all_results

    async def main_loop():
        """Run validation cycles using GRPO evaluation.

        Each cycle:
          1. Fetch miners with models
          2. Test each miner across all required domains
          3. Use GRPO evaluation to determine winner and update scores
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

            # for miner in all_miners_dict.values():
            #     miner.model = "Qwen/Qwen2.5-Coder-32B-Instruct"

            # Keep only miners that have registered a model
            available = [m for m in all_miners_dict.values() if m.model]
            click.echo(f"📋 Found {len(available)} miners with models")
            logger.info(f"Found {len(available)} miners with models")
            
            # Additional filtering by target UIDs if some miners didn't have models
            if target_uids:
                available = [m for m in available if m.uid in target_uids]
                click.echo(f"🎯 Filtered to {len(available)} target miners")
                logger.info(f"Filtered to {len(available)} target miners")
                
            if len(available) < 1:
                click.echo(f"⚠️  No miners available with models.")
                logger.debug("No miners with model; retrying in %ds", delay)
                await asyncio.sleep(delay)
                continue

            # Limit to max batch size
            if len(available) > val_config.MAX_MINERS_PER_BATCH:
                available = available[:val_config.MAX_MINERS_PER_BATCH]
                click.echo(f"🔢 Limited to {len(available)} miners (max batch size)")

            # Run validation cycle across all domains
            all_results = await run_validation_cycle(available)
            
            # Evaluate using GRPO
            click.echo(f"⚡ Computing Evaluation scores...")
            evaluation_round = evaluator.evaluate_round(all_results)
            
            # Show winner
            if evaluation_round.winner_uid is not None:
                click.echo(f"🏆 Winner: UID {evaluation_round.winner_uid} (model: {evaluation_round.winner_model}) with score {evaluation_round.winner_score:.4f}")
            else:
                click.echo("⚠️  No winner determined (insufficient domain coverage or no results)")

            # Show leaderboard
            leaderboard = evaluator.get_leaderboard(limit=10)
            if leaderboard:
                if target_uids:
                    filtered_leaderboard = [perf for perf in leaderboard if perf.uid in target_uids]
                    click.echo(f"\n🏅 GRPO Leaderboard (UIDs: {','.join(map(str, target_uids))}, Envs: {','.join(env_names)}):")
                    display_leaderboard = filtered_leaderboard
                else:
                    click.echo(f"\n🏅 GRPO Leaderboard (Envs: {','.join(env_names)}):")
                    display_leaderboard = leaderboard
                
                click.echo("{:<5} {:<5} {:<30} {:>8} {:>8} {:>8}".format("#", "UID", "Model", "Final", env_names[0] if env_names else "E1", env_names[1] if len(env_names) > 1 else "E2"))
                click.echo("-" * 80)
                
                for rank, perf in enumerate(display_leaderboard[:10], 1):
                    model_display = (perf.model[:28] + "..") if perf.model and len(perf.model) > 30 else (perf.model or "Unknown")
                    score1 = perf.grpo_scores.get(env_names[0], 0.0) if env_names else 0.0
                    score2 = perf.grpo_scores.get(env_names[1], 0.0) if len(env_names) > 1 else 0.0
                    click.echo(f"{rank:<5} {perf.uid:<5} {model_display:<30} {perf.final_score:>8.3f} {score1:>8.3f} {score2:>8.3f}")

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
            logger.info("Starting GRPO validator (s=%d, d=%.1f, envs=%s, uids=%s, cycles=%d)", samples, delay, ','.join(env_names), uids_str, cycles)
        else:
            logger.info("Starting GRPO validator (s=%d, d=%.1f, envs=%s, uids=%s, cycles=∞)", samples, delay, ','.join(env_names), uids_str)
    else:
        if cycles > 0:
            logger.info("Starting GRPO validator (s=%d, d=%.1f, envs=%s, uids=all, cycles=%d)", samples, delay, ','.join(env_names), cycles)
        else:
            logger.info("Starting GRPO validator (s=%d, d=%.1f, envs=%s, uids=all, cycles=∞)", samples, delay, ','.join(env_names))
    
    asyncio.run(main_loop())
    
    # Show final summary if limited cycles were specified
    if cycles > 0:
        click.echo("\n" + "="*60)
        click.echo("🎯 FINAL GRPO VALIDATION SUMMARY")
        click.echo("="*60)
        
        # Show final GRPO leaderboard
        final_leaderboard = evaluator.get_leaderboard(limit=20)
        if final_leaderboard:
            if target_uids:
                filtered_leaderboard = [perf for perf in final_leaderboard if perf.uid in target_uids]
                click.echo(f"Final GRPO Rankings (UIDs: {','.join(map(str, target_uids))}, Envs: {','.join(env_names)}):")
                display_leaderboard = filtered_leaderboard
            else:
                click.echo(f"Final GRPO Rankings (Envs: {','.join(env_names)}):")
                display_leaderboard = final_leaderboard
            
            click.echo("{:<5} {:<5} {:<25} {:>8} {:>8} {:>8}".format("#", "UID", "Model", "Final", env_names[0] if env_names else "E1", env_names[1] if len(env_names) > 1 else "E2"))
            click.echo("-" * 75)
            
            for rank, perf in enumerate(display_leaderboard[:15], 1):
                model_display = (perf.model[:23] + "..") if perf.model and len(perf.model) > 25 else (perf.model or "Unknown")
                score1 = perf.grpo_scores.get(env_names[0], 0.0) if env_names else 0.0
                score2 = perf.grpo_scores.get(env_names[1], 0.0) if len(env_names) > 1 else 0.0
                click.echo(f"{rank:<5} {perf.uid:<5} {model_display:<25} {perf.final_score:>8.3f} {score1:>8.3f} {score2:>8.3f}")
        
        click.echo(f"\n✅ GRPO Validation completed: {cycles} cycles, {','.join(env_names)} environments")
        if target_uids:
            click.echo(f"   Miners tested: {','.join(map(str, target_uids))}")
        click.echo(f"   Samples per environment: {samples}")
        click.echo(f"   EMA alpha: {ema_alpha}")
        click.echo(f"   Skew penalty weight: {skew_penalty}")
        click.echo("\n💾 Evaluation scores saved to ~/.affine/results/evaluator_scores.json")

@cli.command('scores')
@click.option("--uids", "-u", default=None, help="Comma-separated UIDs to show (e.g., '1,2,3,4,5'). If not specified, shows all.")
@click.option("--limit", "-l", default=20, help="Maximum number of miners to show")
def show_scores(uids: str, limit: int):
    """Show current Evaluation Scores."""
    evaluator = Evaluator()
    
    leaderboard = evaluator.get_leaderboard()
    if not leaderboard:
        click.echo("No Evaluation Scores found. Run 'af validate' to generate Evaluation scores.")
        return
    
    # Parse UIDs if provided
    target_uids = None
    if uids:
        try:
            target_uids = [int(x.strip()) for x in uids.split(',')]
        except ValueError:
            click.echo(f"Invalid UIDs format: {uids}. Use comma-separated integers like '1,2,3,4,5'", err=True)
            sys.exit(1)
    
    # Filter leaderboard if UIDs specified
    if target_uids:
        filtered_leaderboard = [perf for perf in leaderboard if perf.uid in target_uids]
        click.echo(f"🏅 Evaluation Scores (UIDs: {','.join(map(str, target_uids))}):")
        display_leaderboard = filtered_leaderboard
    else:
        click.echo("🏅 Evaluation Scores (All):")
        display_leaderboard = leaderboard[:limit]
    
    if not display_leaderboard:
        click.echo("No miners found with the specified UIDs.")
        return
    
    # Determine available domains
    domains = set()
    for perf in display_leaderboard:
        domains.update(perf.grpo_scores.keys())
    domains = sorted(list(domains))
    
    # Header
    header_parts = ["{:<5}", "{:<5}", "{:<25}", "{:>8}"]
    header_values = ["#", "UID", "Model", "Final"]
    
    for domain in domains[:3]:  # Show up to 3 domains
        header_parts.append("{:>8}")
        header_values.append(domain[:7])
    
    header_format = " ".join(header_parts)
    click.echo(header_format.format(*header_values))
    click.echo("-" * (10 + 5 + 25 + 8 + 8 * len(domains[:3])))
    
    for rank, perf in enumerate(display_leaderboard, 1):
        model_display = (perf.model[:23] + "..") if perf.model and len(perf.model) > 25 else (perf.model or "Unknown")
        
        row_values = [rank, perf.uid, model_display, f"{perf.final_score:.3f}"]
        
        for domain in domains[:3]:
            score = perf.grpo_scores.get(domain, 0.0)
            row_values.append(f"{score:.3f}")
        
        row_format = f"{rank:<5} {perf.uid:<5} {model_display:<25} {perf.final_score:>8.3f}"
        for domain in domains[:3]:
            score = perf.grpo_scores.get(domain, 0.0)
            row_format += f" {score:>8.3f}"
        
        click.echo(row_format)
    
    click.echo(f"\nTotal miners with Evaluation Scores: {len(leaderboard)}")
    click.echo(f"Domains: {', '.join(domains)}")
    
    # Show recent rounds
    recent_rounds = evaluator.get_recent_rounds(limit=5)
    if recent_rounds:
        click.echo(f"\n📊 Recent Evaluation Rounds:")
        for round_obj in recent_rounds[-3:]:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(round_obj.timestamp))
            winner_info = f"UID {round_obj.winner_uid}" if round_obj.winner_uid else "No winner"
            click.echo(f"   {timestamp_str}: {winner_info} (score: {round_obj.winner_score:.3f})")

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
