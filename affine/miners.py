import os
import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
from huggingface_hub import HfApi
from .query import _get_client

logger = __import__("logging").getLogger("affine")

MODEL_GATING_CACHE = {}
_GATING_LOCKS: Dict[int, asyncio.Lock] = {}
GATING_TTL = 3600

WEIGHTS_SHA_CACHE: Dict[tuple, tuple] = {}
_WEIGHTS_LOCKS: Dict[int, asyncio.Lock] = {}
WEIGHTS_TTL = 3600

def _get_gating_lock() -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    lock = _GATING_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _GATING_LOCKS[key] = lock
    return lock

def _get_weights_lock() -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    key = id(loop)
    lock = _WEIGHTS_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _WEIGHTS_LOCKS[key] = lock
    return lock

async def check_model_gated(model_id: str, revision: Optional[str] = None) -> Optional[bool]:
    import requests
    async with _get_gating_lock():
        now = time.time()
        cached = MODEL_GATING_CACHE.get(model_id)
        if cached and now - cached[1] < GATING_TTL:
            return cached[0]
        try:
            r = await asyncio.to_thread(requests.get, f"https://huggingface.co/api/models/{model_id}", timeout=5)
            if r.status_code == 200:
                is_gated = r.json().get("gated", False)
                if revision:
                    try:
                        ok = await asyncio.to_thread(lambda: bool(HfApi(token=os.getenv("HF_TOKEN")).repo_info(repo_id=model_id, revision=revision, repo_type="model")))
                        if not ok: is_gated = True
                    except:
                        pass
                MODEL_GATING_CACHE[model_id] = (is_gated, now)
                return is_gated
        except Exception as e:
            logger.trace(f"Gate check failed for {model_id}: {e}")
        if cached:
            MODEL_GATING_CACHE[model_id] = (cached[0], now)
            return cached[0]
        return None

async def get_chute(chutes_id: str) -> Dict:
    url = f"https://api.chutes.ai/chutes/{chutes_id}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    sess = await _get_client()
    async with sess.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as r:
        text = await r.text(errors="ignore")
        if r.status != 200:
            return None
        info = await r.json()
        for k in ('readme','cords','tagline','instances'):
            info.pop(k, None)
        info.get('image', {}).pop('readme', None)
        return info

async def get_chute_code(identifier: str) -> Optional[str]:
    url = f"https://api.chutes.ai/chutes/code/{identifier}"
    token = os.getenv("CHUTES_API_KEY", "")
    headers = {"Authorization": token}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            return await r.text(errors="ignore")

async def get_latest_chute_id(model_name: str, api_key: Optional[str] = None) -> Optional[str]:
    token = api_key or os.getenv("CHUTES_API_KEY", "")
    if not token: return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.chutes.ai/chutes/", headers={"Authorization": token}) as r:
                if r.status != 200: return None
                data = await r.json()
    except Exception: return None
    chutes = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(chutes, list): return None
    for chute in reversed(chutes):
        if any(chute.get(k) == model_name for k in ("model_name", "name", "readme")):
            return chute.get("chute_id")
    return None

async def get_weights_shas(model_id: str, revision: Optional[str] = None) -> Optional[set]:
    key = (model_id, revision)
    now = time.time()
    cached = WEIGHTS_SHA_CACHE.get(key)
    if cached and now - cached[1] < WEIGHTS_TTL:
        return cached[0]
    async with _get_weights_lock():
        cached = WEIGHTS_SHA_CACHE.get(key)
        if cached and now - cached[1] < WEIGHTS_TTL:
            return cached[0]
        try:
            def _repo_info():
                return HfApi(token=os.getenv("HF_TOKEN")).repo_info(
                    repo_id=model_id, repo_type="model", revision=revision, files_metadata=True
                )
            info = await asyncio.to_thread(_repo_info)
            sib = getattr(info, "siblings", None) or []
            def _name(s): return getattr(s, "rfilename", None) or getattr(s, "path", "")
            shas = {str(getattr(s, "lfs", {})["sha256"]) for s in sib
                    if (isinstance(getattr(s, "lfs", None), dict) and _name(s).endswith(".safetensors") and "sha256" in getattr(s, "lfs", {}))}
            WEIGHTS_SHA_CACHE[key] = (shas or None, now)
            return shas or None
        except Exception as e:
            logger.trace(f"HF weights sha lookup failed for {model_id}@{revision}: {e}")
            WEIGHTS_SHA_CACHE[key] = (None, now)
            return None

async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = None,
    meta: object = None,
) -> Dict[int, "Miner"]:
    from . import Miner, NETUID
    from .utils.subtensor import get_subtensor
    
    if netuid is None:
        netuid = NETUID
    
    blacklist_str = os.getenv("AFFINE_MINER_BLACKLIST", "").strip()
    blacklisted_hotkeys = set()
    if blacklist_str:
        blacklisted_hotkeys = {hk.strip() for hk in blacklist_str.split(",") if hk.strip()}
        if blacklisted_hotkeys:
            logger.info(f"Loaded {len(blacklisted_hotkeys)} blacklisted hotkeys from AFFINE_MINER_BLACKLIST")
    
    sub = await get_subtensor()
    meta = meta or await sub.metagraph(netuid)
    commits = await sub.get_all_revealed_commitments(netuid)
    if uids is None:uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int): uids = [uids]
    meta_sem = asyncio.Semaphore(int(os.getenv("AFFINE_META_CONCURRENCY", "12")))
    async def fetch(uid: int):
        try:
            hotkey = meta.hotkeys[ uid ]
            if hotkey in blacklisted_hotkeys:
                logger.debug(f"Skipping blacklisted miner uid={uid} hotkey={hotkey}")
                return None
            if hotkey not in commits: return None
            commit = commits[hotkey]
            block, data = commit[-1]
            block = 0 if uid == 0 else block
            data = json.loads(data)
            model, miner_revision, chute_id = data.get("model"), data.get("revision"), data.get("chute_id")
            async with meta_sem:
                chute = await get_chute(chute_id)
            if not chute: return None
            if not chute.get("hot", False): return None
            gated = await check_model_gated(model)
            if gated is None or gated is True: return None
            chutes_name, slug, chutes_revision = chute.get('name'), chute.get("slug"), chute.get("revision")
            if model != chutes_name or (uid != 0 and chutes_name.split('/')[1].lower()[:6] != 'affine'): return None
            if chutes_revision == None or miner_revision == chutes_revision:
                async with meta_sem:
                    shas = await get_weights_shas(model, miner_revision)
                return Miner(
                    uid=uid, hotkey=hotkey, model=model, block=int(block),
                    revision = miner_revision,
                    slug = slug,
                    chute=chute,
                    weights_shas=shas,
                )
        except: pass
    results = await asyncio.gather(*(fetch(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}
    if output:
        earliest_by_sha: Dict[str, tuple] = {}
        for uid, m in output.items():
            if not m.weights_shas: continue
            blk = m.block if isinstance(m.block, int) else (int(m.block) if m.block is not None else (2**63 - 1))
            for sha in m.weights_shas:
                prev = earliest_by_sha.get(sha)
                if prev is None or blk < prev[0]:
                    earliest_by_sha[sha] = (blk, uid)
        if earliest_by_sha:
            keep = set(output.keys())
            for uid, m in output.items():
                if m.weights_shas and any(earliest_by_sha.get(s, (None, uid))[1] != uid for s in m.weights_shas):
                    if uid in keep: keep.remove(uid)
            output = {uid: m for uid, m in output.items() if uid in keep}
        if output:
            best_by_model: Dict[str, tuple] = {}
            for uid, m in output.items():
                if not m.model:
                    continue
                blk = m.block if isinstance(m.block, int) else (int(m.block) if m.block is not None else (2**63 - 1))
                prev = best_by_model.get(m.model)
                if prev is None or blk < prev[0]:
                    best_by_model[m.model] = (blk, uid)
            selected_uids = {uid for _, uid in best_by_model.values()}
            output = {uid: m for uid, m in output.items() if uid in selected_uids}
    return output