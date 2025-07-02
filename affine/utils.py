"""
Affine validation module - handles model validation including hot status checks
"""
import asyncio, aiohttp
from typing import Dict, Any, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from . import Miner
else:
    Miner = None


async def get_chutes_info(model: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
    """Fetches additional information about a model from the Chutes.ai API."""
    from affine import get_conf
    key = get_conf("CHUTES_API_KEY")
    if not key:
        return None
    url = f"https://api.chutes.ai/chutes/{model.replace('/','_')}"
    try:
        async with session.get(url, headers={"Authorization": key}) as r:
            return await r.json() if r.status == 200 else None
    except Exception:
        return None


async def is_model_hot(model: str, session: aiohttp.ClientSession) -> bool:
    info = await get_chutes_info(model, session)
    return bool(info and info.get("hot"))


async def validate_miners_hot(miners_dict: Dict[int, Miner], require_hot: bool = True) -> Dict[int, Miner]:
    """Return miners whose model is ‘hot’ (or all if require_hot=False)."""
    if not miners_dict:
        return miners_dict
    async with aiohttp.ClientSession() as sess:
        checks = await asyncio.gather(
            *[is_model_hot(m.model, sess) if m.model else True for m in miners_dict.values()],
            return_exceptions=True
        )
    good = lambda ok, m: (not require_hot) or (not m.model) or (isinstance(ok, bool) and ok)
    return {uid: m for (uid, m), ok in zip(miners_dict.items(), checks) if good(ok, m)}


async def validate_single_model_hot(model: str) -> bool:
    """Convenience function to check if a single model is hot."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        return await is_model_hot(model, session)


async def get_chutes_model_info(model: str) -> dict:
    """Fetches vLLM config for a model from the Chutes.ai API."""
    from affine import get_conf
    api_key = get_conf("CHUTES_API_KEY")
    if not api_key:
        print(f"No Chutes API key available for model info fetch")
        return None
    url = f"https://api.chutes.ai/guess/vllm_config?model={model}"
    headers = {"Authorization": api_key}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"vLLM config fetched for model '{model}'")
                    return data
                else:
                    print(f"Failed to fetch vLLM config: HTTP {response.status}")
                    return None
    except Exception as e:
        print(f"Error fetching vLLM config: {e}")
        return None

async def get_model_size(model: str) -> dict:
    chutes_data = await get_chutes_model_info(model)

    model_size = chutes_data['total_model_size']

    return model_size


