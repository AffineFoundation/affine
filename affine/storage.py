import os
import json
import time
import asyncio
import logging
import aiohttp
import orjson
import aiofiles
from pathlib import Path
from typing import AsyncIterator
from tqdm.asyncio import tqdm

from aiobotocore.session import get_session
from botocore.config import Config

from affine.config import get_conf
from affine.utils.subtensor import get_subtensor
from affine.models import Result
from affine.query import _get_client
from affine.setup import logger

WINDOW        = int(os.getenv("AFFINE_WINDOW", 20))
RESULT_PREFIX = "affine/results/"
INDEX_KEY     = "affine/index.json"

FOLDER  = os.getenv("R2_FOLDER", "affine" )
BUCKET  = os.getenv("R2_BUCKET_ID", "00523074f51300584834607253cae0fa" )
ACCESS  = os.getenv("R2_WRITE_ACCESS_KEY_ID", "")
SECRET  = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

PUBLIC_READ = os.getenv("AFFINE_R2_PUBLIC", "1") == "1"
ACCOUNT_ID  = os.getenv("R2_ACCOUNT_ID", BUCKET)

R2_PUBLIC_BASE = f"https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev"

get_client_ctx = lambda: get_session().create_client(
    "s3", endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS, aws_secret_access_key=SECRET,
    config=Config(max_pool_connections=256)
)

CACHE_DIR = Path(os.getenv("AFFINE_CACHE_DIR",
                 Path.home() / ".cache" / "affine" / "blocks"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _w(b: int) -> int: return (b // WINDOW) * WINDOW

async def _index() -> list[str]:
    if PUBLIC_READ:
        sess = await _get_client()
        url = f"{R2_PUBLIC_BASE}/{INDEX_KEY}"
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            r.raise_for_status()
            return json.loads(await r.text())
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
                               Body=orjson.dumps(sorted(idx)),
                               ContentType="application/json")

async def _cache_shard(key: str, sem: asyncio.Semaphore) -> Path:
    name, out = Path(key).name, None
    out = CACHE_DIR / f"{name}.jsonl"; mod = out.with_suffix(".modified")
    async with sem:
        if PUBLIC_READ:
            sess = await _get_client()
            url = f"{R2_PUBLIC_BASE}/{key}"
            async with sess.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                r.raise_for_status()
                body = await r.read()
                lm = r.headers.get("last-modified", str(time.time()))
            tmp = out.with_suffix(".tmp")
            with tmp.open("wb") as f:
                f.write(b"\n".join(orjson.dumps(i) for i in orjson.loads(body)) + b"\n")
            os.replace(tmp, out); mod.write_text(lm)
            return out
        async with get_client_ctx() as c:
            if out.exists() and mod.exists():
                h = await c.head_object(Bucket=FOLDER, Key=key)
                if h["LastModified"].isoformat() == mod.read_text().strip():
                    return out
            o = await c.get_object(Bucket=FOLDER, Key=key)
            body, lm = await o["Body"].read(), o["LastModified"].isoformat()
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f:
        f.write(b"\n".join(orjson.dumps(i) for i in orjson.loads(body)) + b"\n")
    os.replace(tmp, out); mod.write_text(lm)
    return out

async def _jsonl(path: Path) -> AsyncIterator[bytes]:
    async with aiofiles.open(path, "rb") as f:
        async for line in f:
            yield line.rstrip(b"\n")

async def dataset(
    tail: int,
    *,
    max_concurrency: int = 10,
) -> AsyncIterator["Result"]:
    sub  = await get_subtensor()
    cur  = await sub.get_current_block()
    need = {w for w in range(_w(cur - tail), _w(cur) + WINDOW, WINDOW)}
    keys = [
        k for k in await _index()
        if (h := Path(k).name.split("-", 1)[0]).isdigit() and int(h) in need
    ]
    keys.sort()    
    sem = asyncio.Semaphore(max_concurrency)
    async def _prefetch(key: str) -> Path:
        return await _cache_shard(key, sem)
    tasks: list[asyncio.Task[Path]] = [
        asyncio.create_task(_prefetch(k)) for k in keys[:max_concurrency]
    ]
    next_key = max_concurrency
    bar = tqdm(desc=f"Dataset=({cur}, {cur - tail})", unit="res", dynamic_ncols=True)
    try:
        for i, key in enumerate(keys):
            try:
                path = await tasks[i]
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"task key: {key}, failed, skip")
                continue
            if next_key < len(keys):
                tasks.append(asyncio.create_task(_prefetch(keys[next_key])))
                next_key += 1
            async for raw in _jsonl(path):
                try:
                    r = Result.model_validate(orjson.loads(raw))
                    if r.verify():
                        bar.update(1)
                        yield r
                except Exception:
                    pass
    finally:
        bar.close()

async def sign_results( wallet, results ):
    try:
        signer_url = get_conf('SIGNER_URL', default='http://signer:8080')
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
        logger.info(f"sink: signer unavailable, using local signing: {type(e).__name__}: {e}")
        hotkey = wallet.hotkey.ss58_address
        for r in results: 
            r.sign(wallet)
    finally:
        return hotkey, results

SINK_BUFFER: list["Result"] = []
SINK_BUFFER_SIZE = int(os.getenv("AFFINE_SINK_BUFFER", "100"))

async def sink_enqueue(wallet, block, results, force: bool = False):
    global SINK_BUFFER
    SINK_BUFFER.extend(results)
    if not force and len(SINK_BUFFER) < SINK_BUFFER_SIZE: return
    buf, SINK_BUFFER = SINK_BUFFER, []
    await sink(wallet=wallet, results=buf, block=block)

async def sink(wallet, results: list["Result"], block: int = None):
    if not results: return
    if block is None:
        sub = await get_subtensor(); block = await sub.get_current_block()
    hotkey, signed = await sign_results(wallet, results)
    timestamp = time.time_ns()  # ensure each upload gets a unique key
    key = f"{RESULT_PREFIX}{_w(block):09d}-{hotkey}-{timestamp}.json"
    dumped = [r.model_dump(mode="json") for r in signed]
    async with get_client_ctx() as c:
        await c.put_object(
            Bucket=FOLDER,
            Key=key,
            Body=orjson.dumps(dumped),
            ContentType="application/json",
        )
    await _update_index(key)

async def prune(tail: int):
    sub = await get_subtensor(); cur = await sub.get_current_block()
    for f in CACHE_DIR.glob("*.jsonl"):
        b = f.name.split("-", 1)[0]
        if b.isdigit() and int(b) < cur - tail:
            try: f.unlink()
            except OSError: pass
