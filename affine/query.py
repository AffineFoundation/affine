import os
import time
import random
import atexit
import asyncio
import logging
import aiohttp
import traceback
from typing import Dict, List, Optional
from affine.config import get_conf
from affine.models import Response, Result, Evaluation, Miner, Challenge
from affine.tasks import BaseSDKEnv
from affine.setup import logger


_HTTP_SEMS: Dict[int, asyncio.Semaphore] = {}
_CLIENTS: Dict[int, aiohttp.ClientSession] = {}

async def _cleanup_clients():
    for client in _CLIENTS.values():
        if client and not client.closed:
            await client.close()
    _CLIENTS.clear()

def _sync_cleanup():
    try:
        asyncio.run(_cleanup_clients())
    except RuntimeError:
        pass

atexit.register(_sync_cleanup)

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
        limit = int(os.getenv("AFFINE_HTTP_CONCURRENCY", "400"))
        conn = aiohttp.TCPConnector(
            limit=limit,
            limit_per_host=0,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        client = aiohttp.ClientSession(
            connector=conn,
            timeout=aiohttp.ClientTimeout(total=None)
        )
        _CLIENTS[key] = client
    return client

TERMINAL = {400, 404, 410}

async def query(prompt, model: str = "unsloth/gemma-3-12b-it", slug: str = "llm", timeout=150, retries=0, backoff=1):
    url = f"https://{slug}.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {get_conf('CHUTES_API_KEY')}", "Content-Type": "application/json"}
    start = time.monotonic()
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


async def query_miner(env: BaseSDKEnv, miner: Miner, task_id: int) -> Result:
    """
    Query a miner using SDK environment interface.
    
    Args:
        env: SDK environment instance (e.g., DED(), ALFWORLD())
        miner: Miner instance to query
        task_id: Optional task ID for environments that support it
    
    Returns:
        Result object containing evaluation outcome
    """
    if not miner.model:
        logger.warning(f"Miner uid={miner.uid} has no model, skipping")
        return None
    
    logger.trace(f"Querying miner uid={miner.uid} on env={env.env_name} task_id={task_id}")
    
    start = time.monotonic()
    
    try:
        # Call SDK evaluate method
        evaluation_result = await env.evaluate(miner, task_id=task_id)
        
        # Build response
        response = Response(
            response=None,
            latency_seconds=time.monotonic() - start,
            attempts=1,
            model=miner.model,
            error=None,
            success=evaluation_result.extra.get("success", False)
        )
        
        # Build challenge from env
        challenge = Challenge(
            env=env.env_name,
            prompt=f"{env.env_name} placeholder",
            extra={"task_id": task_id},
            challenge_id=f"{env.env_name}:{task_id}"
        )
        
        # Build evaluation
        evaluation = Evaluation(
            env=env.env_name,
            score=evaluation_result.score,
            extra=evaluation_result.extra
        )
        
        return Result(
            miner=miner,
            challenge=challenge,
            response=response,
            evaluation=evaluation
        )
        
    except Exception as e:
        response = Response(
            response=None,
            latency_seconds=time.monotonic() - start,
            attempts=1,
            model=miner.model or "",
            error=str(e),
            success=False
        )
        
        challenge = Challenge(
            env=env.env_name,
            prompt=f"{env.env_name} placeholder",
            extra={"task_id": task_id, "error": str(e)},
            challenge_id=f"{env.env_name}:{task_id}"
        )
        
        evaluation = Evaluation(
            env=env.env_name,
            score=0.0,
            extra={"error": str(e), "evaluation_failed": True}
        )
        
        return Result(
            miner=miner,
            challenge=challenge,
            response=response,
            evaluation=evaluation
        )