import os
import json
import time
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Union
from huggingface_hub import HfApi
from affine.query import _get_client
from affine.models import Miner
from affine.setup import NETUID
from affine.utils.subtensor import get_subtensor

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


async def check_model_gated(
    model_id: str, revision: Optional[str] = None
) -> Optional[bool]:
    async with _get_gating_lock():
        now = time.time()
        cached = MODEL_GATING_CACHE.get(model_id)
        if cached and now - cached[1] < GATING_TTL:
            return cached[0]
        try:
            r = await asyncio.to_thread(
                requests.get, f"https://huggingface.co/api/models/{model_id}", timeout=5
            )
            if r.status_code == 200:
                is_gated = r.json().get("gated", False)
                if revision:
                    try:
                        ok = await asyncio.to_thread(
                            lambda: bool(
                                HfApi(token=os.getenv("HF_TOKEN")).repo_info(
                                    repo_id=model_id,
                                    revision=revision,
                                    repo_type="model",
                                )
                            )
                        )
                        if not ok:
                            is_gated = True
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
    async with sess.get(
        url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
    ) as r:
        text = await r.text(errors="ignore")
        if r.status != 200:
            return None
        info = await r.json()
        for k in ("readme", "cords", "tagline", "instances"):
            info.pop(k, None)
        info.get("image", {}).pop("readme", None)
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


async def get_latest_chute_id(
    model_name: str, api_key: Optional[str] = None
) -> Optional[str]:
    token = api_key or os.getenv("CHUTES_API_KEY", "")
    if not token:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.chutes.ai/chutes/", headers={"Authorization": token}
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
    except Exception:
        return None
    chutes = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(chutes, list):
        return None
    for chute in reversed(chutes):
        if any(chute.get(k) == model_name for k in ("model_name", "name", "readme")):
            return chute.get("chute_id")
    return None


async def get_weights_shas(
    model_id: str, revision: Optional[str] = None
) -> Optional[set]:
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
                    repo_id=model_id,
                    repo_type="model",
                    revision=revision,
                    files_metadata=True,
                )

            info = await asyncio.to_thread(_repo_info)
            sib = getattr(info, "siblings", None) or []

            def _name(s):
                return getattr(s, "rfilename", None) or getattr(s, "path", "")

            shas = {
                str(getattr(s, "lfs", {})["sha256"])
                for s in sib
                if (
                    isinstance(getattr(s, "lfs", None), dict)
                    and _name(s).endswith(".safetensors")
                    and "sha256" in getattr(s, "lfs", {})
                )
            }
            WEIGHTS_SHA_CACHE[key] = (shas or None, now)
            return shas or None
        except Exception as e:
            logger.trace(f"HF weights sha lookup failed for {model_id}@{revision}: {e}")
            WEIGHTS_SHA_CACHE[key] = (None, now)
            return None


def _normalize_block(block) -> int:
    if isinstance(block, int):
        return block
    return int(block) if block is not None else (2**63 - 1)


def _load_blacklist() -> set:
    blacklist_str = os.getenv("AFFINE_MINER_BLACKLIST", "").strip()
    if not blacklist_str:
        return set()
    hotkeys = {hk.strip() for hk in blacklist_str.split(",") if hk.strip()}
    if hotkeys:
        logger.info(
            f"Loaded {len(hotkeys)} blacklisted hotkeys from AFFINE_MINER_BLACKLIST"
        )
    return hotkeys


def _filter_by_earliest_sha(output: Dict[int, "Miner"]) -> Dict[int, "Miner"]:
    earliest_by_sha: Dict[str, tuple] = {}
    for uid, m in output.items():
        if not m.weights_shas:
            continue
        block = _normalize_block(m.block)
        for sha in m.weights_shas:
            prev = earliest_by_sha.get(sha)
            if prev is None or block < prev[0]:
                earliest_by_sha[sha] = (block, uid)

    if not earliest_by_sha:
        return output

    keep = set(output.keys())
    for uid, m in output.items():
        if m.weights_shas:
            for sha in m.weights_shas:
                if earliest_by_sha.get(sha, (None, uid))[1] != uid:
                    keep.discard(uid)
                    break

    return {uid: m for uid, m in output.items() if uid in keep}


def _filter_by_best_model(output: Dict[int, "Miner"]) -> Dict[int, "Miner"]:
    best_by_model: Dict[str, tuple] = {}
    for uid, m in output.items():
        if not m.model:
            continue
        block = _normalize_block(m.block)
        prev = best_by_model.get(m.model)
        if prev is None or block < prev[0]:
            best_by_model[m.model] = (block, uid)

    selected_uids = {uid for _, uid in best_by_model.values()}
    return {uid: m for uid, m in output.items() if uid in selected_uids}


async def miners(
    uids: Optional[Union[int, List[int]]] = None,
    netuid: int = NETUID,
    meta: object = None,
    check_validity: bool = True,
) -> Dict[int, "Miner"]:
    blacklisted_hotkeys = _load_blacklist()

    sub = await get_subtensor()
    meta = meta or await sub.metagraph(netuid)
    commits = await sub.get_all_revealed_commitments(netuid)
    if uids is None:
        uids = list(range(len(meta.hotkeys)))
    elif isinstance(uids, int):
        uids = [uids]
    meta_sem = asyncio.Semaphore(int(os.getenv("AFFINE_META_CONCURRENCY", "12")))

    async def _fetch_miner(uid: int) -> Optional["Miner"]:
        try:
            hotkey = meta.hotkeys[uid]
            if hotkey in blacklisted_hotkeys:
                logger.debug(f"Skipping blacklisted miner uid={uid} hotkey={hotkey}")
                return None
            if hotkey not in commits:
                return None

            block, commit_data = commits[hotkey][-1]
            block = 0 if uid == 0 else block
            data = json.loads(commit_data)
            model = data.get("model")

            miner_revision = data.get("revision")
            chute_id = data.get("chute_id")

            gated = await check_model_gated(model)
            if gated is None or gated:
                return None

            async with meta_sem:
                weights_shas = await get_weights_shas(model, miner_revision)

            if not check_validity:
                return Miner(
                    uid=uid,
                    hotkey=hotkey,
                    model=model,
                    block=int(block),
                    revision=miner_revision,
                    weights_shas=weights_shas,
                )

            async with meta_sem:
                chute = await get_chute(chute_id)

            if not chute or not chute.get("hot", False):
                return None

            chute_name = chute.get("name")
            if model != chute_name:
                return None

            if uid != 0 and chute_name.split("/")[1].lower()[:6] != "affine":
                return None

            chute_revision = chute.get("revision")
            if chute_revision is not None and miner_revision != chute_revision:
                return None

            return Miner(
                uid=uid,
                hotkey=hotkey,
                model=model,
                block=int(block),
                revision=miner_revision,
                slug=chute.get("slug"),
                chute=chute,
                weights_shas=weights_shas,
            )
        except Exception as e:
            logger.trace(f"Failed to fetch miner uid={uid}: {e}")
            return None

    results = await asyncio.gather(*(_fetch_miner(uid) for uid in uids))
    output = {uid: m for uid, m in zip(uids, results) if m is not None}

    if not output:
        return output

    output = _filter_by_earliest_sha(output)
    if output:
        output = _filter_by_best_model(output)

    return output
