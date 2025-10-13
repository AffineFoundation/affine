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
from affine.models import Response, Result, Evaluation, Miner, ContainerEnv
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

LOG_TEMPLATE = (
    "[RESULT] "
    "{pct:>3.0f}% | "
    "U{uid:>3d} │ "
    "{model:<50s} │ "
    "{env:<20} │ "
    "{success:^4s} │ "
    "{score:>6.4f} │ "
    "{latency:>6.3f}s"
)

async def run(challenges, miners, timeout=240, retries=0, backoff=1, task_ids: Optional[Dict[str, int]] = None)-> List:
    if not isinstance(challenges, list): challenges = [challenges]
    if isinstance(miners, Miner): miners = [miners]
    if isinstance(miners, dict):  mmap = miners
    elif isinstance(miners, list) and all(hasattr(m, "uid") for m in miners):  mmap = {m.uid: m for m in miners}
    else: mmap = await miners(miners)
    
    logger.trace("Running challenges: %s on miners: %s", [chal.prompt[:30] for chal in challenges], list(mmap.keys()))
    response = []
    
    async def proc(miner, chal):
        if isinstance(chal.env, ContainerEnv):
            start = time.monotonic()
            try:
                ev = await chal.env.run_episode(policy=miner, task_id=(task_ids or {}).get(chal.env.name) if task_ids else None)
                resp = Response(response=None, latency_seconds=time.monotonic()-start, attempts=1, model=miner.model or "", error=None, success=True)
            except Exception as e:
                traceback.print_exc()
                ev = Evaluation(env=chal.env, score=0.0, extra={"error": str(e), "evaluation_failed": True})
                resp = Response(response=None, latency_seconds=time.monotonic()-start, attempts=1, model=miner.model or "", error=str(e), success=False)
            logger.info(f"[SCORE] U{miner.uid:>3d} {chal.env.name:<20} = {ev.score:.4f}")
            return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)
        start = time.monotonic()
        err = f"Unsupported environment type: {type(chal.env).__name__}"
        ev = Evaluation(env=chal.env, score=0.0, extra={"error": err, "evaluation_failed": True})
        resp = Response(response=None, latency_seconds=time.monotonic()-start, attempts=1, model=miner.model or "", error=err, success=False)
        return Result(miner=miner, challenge=chal, response=resp, evaluation=ev)
    
    tasks = [ asyncio.create_task(proc(m, chal)) for m in mmap.values() if m.model for chal in challenges]  
    total = len(tasks); completed = 0
    for task in asyncio.as_completed(tasks): 
        result = await task
        response.append(result); completed += 1
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
    return response