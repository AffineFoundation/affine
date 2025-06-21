"""
Core data models and functions for the Affine framework.
This module contains the essential blockchain mining functionality.
"""

import os
import time
import random
import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import bittensor as bt
from alive_progress import alive_bar

from .config import config

# Setup logger
logger = logging.getLogger("affine")


class Miner(BaseModel):
    """Represents a miner in the Bittensor network."""
    uid: int
    hotkey: str
    model: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None


class Response(BaseModel):
    """Response from an LLM model."""
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]


class BaseEnv(BaseModel, ABC):
    """Abstract base class for challenge environments."""
    
    class Config:
        arbitrary_types_allowed = True

    async def many(self, n: int) -> List["Challenge"]:
        """Generate multiple challenges."""
        return [await self.generate() for _ in range(n)]

    @abstractmethod
    async def generate(self) -> "Challenge":
        """Generate a single challenge."""
        ...

    @abstractmethod
    async def evaluate(self, challenge: "Challenge", response: Response) -> "Evaluation":
        """Evaluate a response to a challenge."""
        ...


class Challenge(BaseModel):
    """A challenge to be solved by an LLM."""
    env: BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)

    async def evaluate(self, response: Response) -> "Evaluation":
        """Evaluate this challenge against a response."""
        return await self.env.evaluate(self, response)


class Evaluation(BaseModel):
    """Evaluation result of a challenge-response pair."""
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    """Complete result combining miner, challenge, response, and evaluation."""
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation


async def get_chute(model: str) -> Dict[str, Any]:
    """Get chute information for a model from the Chutes.ai API."""
    chutes_username = config.get("CHUTES_USER", "default")
    api_url = f"https://api.chutes.ai/chutes/{chutes_username}/{model}"
    token = config.get_required("CHUTES_API_KEY")
    headers = {"Authorization": token}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url, headers=headers) as response:
            if response.status != 200:
                text = await response.text(errors="ignore")
                raise RuntimeError(f"API error {response.status}: {text}")
            
            chute_info = await response.json()
            # Clean up unnecessary fields
            for field in ['readme', 'chords', 'tagline', 'instances']:
                chute_info.pop(field, None)
            chute_info.get('image', {}).pop('readme', None)
            
            logger.trace("Fetched chute info for %s", model)
            return chute_info


async def miners(
    uids: Optional[Union[int, List[int]]] = None, 
    no_null: bool = False
) -> Dict[int, Miner]:
    """
    Get miners from the Bittensor network.
    
    Args:
        uids: Specific UIDs to fetch, or None for all
        no_null: If True, filter out miners without models/blocks
    """
    NETUID = 120
    subtensor = bt.async_subtensor()
    await subtensor.initialize()
    
    meta = await subtensor.metagraph(NETUID)
    revs = await subtensor.get_all_revealed_commitments(NETUID)

    # Normalize UIDs input
    if uids is None:
        uids_list = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids_list = [uids]
    else:
        uids_list = uids

    # Build miners dict
    miners_dict: Dict[int, Miner] = {}
    for uid in uids_list:
        if 0 <= uid < len(meta.hotkeys):
            hotkey = meta.hotkeys[uid]
            commits = revs.get(hotkey, [])
            block, model = commits[-1] if commits else (None, None)
            
            # Apply no_null filter
            if no_null and block is None:
                continue
            
            miners_dict[uid] = Miner(
                uid=uid,
                hotkey=hotkey,
                model=str(model) if model is not None else None,
                block=int(block) if block is not None else None
            )

    # Fetch chute information for miners with models
    miners_with_models = [m for m in miners_dict.values() if m.model]
    if miners_with_models:
        chute_tasks = [get_chute(m.model) for m in miners_with_models]
        chute_results = await asyncio.gather(*chute_tasks, return_exceptions=True)
        
        for miner, chute_info in zip(miners_with_models, chute_results):
            if not isinstance(chute_info, Exception):
                miner.chute = chute_info

    # Enhanced logging
    miner_info = [
        f"\tUID: {m.uid}, Hotkey: {m.hotkey[:8]}..., Model: {m.model}, Block: {m.block}"
        for m in miners_dict.values()
    ]
    logger.debug("Discovered %d miners:\n%s", len(miners_dict), "\n".join(miner_info))

    return miners_dict


async def _run_single_challenge(
    session: aiohttp.ClientSession,
    challenge: Challenge,
    model: str,
    timeout: float,
    retries: int,
    backoff: float
) -> Response:
    """Run a single challenge against a model with exponential backoff."""
    api_url = "https://llm.chutes.ai/v1/chat/completions"
    token = config.get_required("CHUTES_API_KEY")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": challenge.prompt}]
    }
    
    start_time = time.monotonic()
    
    for attempt in range(1, retries + 2):
        try:
            async with session.post(
                api_url, json=payload, headers=headers, timeout=timeout
            ) as response:
                if response.status != 200:
                    text = await response.text(errors="ignore")
                    raise RuntimeError(f"HTTP {response.status}: {text}")
                
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                latency = time.monotonic() - start_time
                
                logger.trace("Model %s answered in %.2fs on attempt %d", model, latency, attempt)
                
                return Response(
                    response=content,
                    latency_seconds=latency,
                    attempts=attempt,
                    model=model,
                    error=None
                )
                
        except Exception as e:
            logger.debug("Attempt %d for %s failed: %s", attempt, model, e)
            if attempt > retries:
                return Response(
                    response=None,
                    latency_seconds=time.monotonic() - start_time,
                    attempts=attempt,
                    model=model,
                    error=str(e)
                )
            
            # Exponential backoff with jitter
            delay = backoff * (2 ** (attempt - 1)) * (1 + random.uniform(-0.1, 0.1))
            await asyncio.sleep(delay)
    
    # This should never be reached
    return Response(
        response=None,
        latency_seconds=time.monotonic() - start_time,
        attempts=retries + 1,
        model=model,
        error="Maximum retries exceeded"
    )


async def run(
    challenges: Union[Challenge, List[Challenge]],
    miners_param: Optional[Union[Dict[int, Miner], List[Miner], int, List[int]]] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.0,
    show_progress: bool = True
) -> List[Result]:
    """
    Run challenges against miners and return evaluation results.
    
    Args:
        challenges: Challenge(s) to run
        miners_param: Miners to test against
        timeout: Request timeout in seconds
        retries: Number of retries for failed requests
        backoff: Backoff multiplier for retries
        show_progress: Whether to show progress bar
    """
    # Normalize challenges input
    challenges_list = [challenges] if not isinstance(challenges, list) else challenges
    
    # Get miners
    if isinstance(miners_param, dict):
        miners_dict = miners_param
    elif isinstance(miners_param, list) and all(isinstance(m, Miner) for m in miners_param):
        miners_dict = {m.uid: m for m in miners_param}
    else:
        miners_dict = await miners(miners_param)
    
    # Filter miners with valid models
    active_miners = [miner for miner in miners_dict.values() if miner.model]
    
    if not active_miners:
        logger.warning("No active miners found")
        return []
    
    total_tasks = len(active_miners) * len(challenges_list)
    results: List[Result] = []
    
    logger.info("Running %d tasks (%d miners × %d challenges)", 
               total_tasks, len(active_miners), len(challenges_list))
    
    # Run all challenge-miner combinations with progress bar
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        async def run_single_task(miner: Miner, challenge: Challenge) -> Result:
            response = await _run_single_challenge(
                session, challenge, miner.model, timeout, retries, backoff
            )
            evaluation = await challenge.evaluate(response)
            return Result(
                miner=miner,
                challenge=challenge,
                response=response,
                evaluation=evaluation
            )
        
        # Create all tasks
        tasks = [
            asyncio.create_task(run_single_task(miner, challenge))
            for miner in active_miners
            for challenge in challenges_list
        ]
        
        # Execute with progress bar
        if show_progress:
            with alive_bar(total_tasks, title='🚀 Affine') as bar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    bar()
        else:
            results = await asyncio.gather(*tasks)
    
    logger.info("Completed %d evaluations", len(results))
    return results


# Legacy function for backward compatibility
def get_conf(key: str) -> str:
    """Get required configuration value (legacy function)."""
    return config.get_required(key) 