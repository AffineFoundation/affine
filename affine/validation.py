"""
Validation module for the Affine framework.
Handles model validation and Chutes.ai API interactions.
"""

import asyncio
import aiohttp
import os
from typing import Dict, Any, Optional, List
from .config import config


async def get_chutes_info(model: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
    """Fetches additional information about a model from the Chutes.ai API."""
    api_key = config.get("CHUTES_API_KEY")
    if not api_key:
        print(f" No Chutes API key available for model info fetch")
        return None

    # Use the same URL pattern as existing get_chute function
    chutes_username = config.get("CHUTES_USER", "default")
    url = f"https://api.chutes.ai/chutes/{chutes_username}/{model}"
    headers = {"Authorization": api_key}

    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                print(f"✓ Fetched info for model '{model}'")
                return data
            elif response.status == 404:
                print(f" Model '{model}' not found on Chutes")
                return None
            else:
                response_text = await response.text()
                print(f" Failed to fetch info for model {model}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f" Error fetching info for model {model}: {e}")
        return None


async def is_model_hot(model: str, session: aiohttp.ClientSession) -> bool:
    """Checks if a model is marked as 'hot' on Chutes.ai by fetching its info."""
    chutes_info = await get_chutes_info(model, session)
    if chutes_info and isinstance(chutes_info, dict):
        is_hot = chutes_info.get("hot", False)
        print(f" Model '{model}' hot status: {is_hot}")
        return is_hot
    else:
        print(f"Could not verify hot status for model '{model}'. Defaulting to False.")
        return False


async def validate_model_availability(model: str) -> bool:
    """Validates if a model is available and accessible on Chutes.ai."""
    async with aiohttp.ClientSession() as session:
        chutes_info = await get_chutes_info(model, session)
        if chutes_info:
            status = chutes_info.get("status", "unknown")
            if status == "running":
                print(f"Model '{model}' is running and available")
                return True
            else:
                print(f"Model '{model}' status: {status}")
                return False
        return False


async def test_model_response(model: str, test_prompt: str = "Hello, how are you?") -> bool:
    """Tests if a model can respond to a simple prompt."""
    api_key = config.get("CHUTES_API_KEY")
    if not api_key:
        print("No Chutes API key available for model testing")
        return False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": test_prompt}],
        "stream": False,
        "max_tokens": 50,
        "temperature": 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://llm.chutes.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        print(f"Model '{model}' responded successfully")
                        return True
                    else:
                        print(f"Model '{model}' returned empty response")
                        return False
                else:
                    text = await response.text()
                    print(f"Model '{model}' test failed: HTTP {response.status} - {text}")
                    return False
    except Exception as e:
        print(f"Error testing model '{model}': {e}")
        return False


async def validate_models(models: List[str]) -> Dict[str, bool]:
    """Validates multiple models concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [validate_model_availability(model) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        validation_results = {}
        for model, result in zip(models, results):
            if isinstance(result, Exception):
                print(f"Exception validating model '{model}': {result}")
                validation_results[model] = False
            else:
                validation_results[model] = result
        
        return validation_results


async def get_model_stats(model: str) -> Optional[Dict[str, Any]]:
    """Gets detailed statistics for a model from Chutes.ai."""
    async with aiohttp.ClientSession() as session:
        chutes_info = await get_chutes_info(model, session)
        if chutes_info:
            stats = {
                "status": chutes_info.get("status", "unknown"),
                "hot": chutes_info.get("hot", False),
                "gpu_count": chutes_info.get("gpu_count", 0),
                "concurrency": chutes_info.get("concurrency", 0),
                "image": chutes_info.get("image", "unknown"),
                "created_at": chutes_info.get("created_at"),
                "updated_at": chutes_info.get("updated_at")
            }
            return stats
        return None


