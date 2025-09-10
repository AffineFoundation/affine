from __future__ import annotations
import re
import os
import json
import time
import random
import aiohttp
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Literal, TypeVar, Awaitable
from fastapi import FastAPI
from pydantic import BaseModel

__version__: str = "0.0.0"

app = FastAPI(
    title="PROXY",
    description="Affine env proxy.",
)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY", "")
if not CHUTES_API_KEY:
    print("Warning: CHUTES_API_KEY not set. Some features may not work.")
MODEL = 'GANGodfather/Affine-5ER7L69dC9dmuyJ5AT7HxeSYgKjAH5Y7FPy3BLJQmSBH7Zh3'
SLUG = 'gangodfather-gangodfather-affine-5er7l69dc9dmuyj5at7hxesyg'

@app.post("/chutes")
async def chutes(model: str, slug: str, prompt: str, timeout: float = 60) -> dict:
    url = f"https://{slug}.chutes.ai/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {os.environ.get('CHUTES_API_KEY', '')}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    conn = aiohttp.TCPConnector(
        limit=100, 
        limit_per_host=0,
        ttl_dns_cache=300,
        enable_cleanup_closed=True
    )
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(
        connector=conn,
        timeout=timeout_cfg
    ) as client:
        async with client.post(url, json=payload, headers=hdr) as r:
            return await r.json()

class AddRequest(BaseModel):
    x: float
    y: float


@app.post("/run")
async def run(req: AddRequest) -> dict:
    x = req.x
    y = req.y
    prompt = f"{x} + {y} = ?"
    result = await chutes(model=MODEL, slug=SLUG, prompt=prompt)
    response = (result.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    expected = x + y
    z = None
    try:
        m = re.search(r"-?\d+(?:\.\d+)?", response)
        z = float(m.group(0)) if m else None
    except Exception:
        z = None
    score = 1.0 if (z is not None and abs(z - expected) < 1e-9) else 0.0
    return {"x": x, "y": y, "prompt": prompt, "response": response, "z": z, "expected": expected, "score": score}

    
async def main():
    import quixand as qs
    docker_file = os.path.dirname(__file__) + "/"
    print(docker_file)
    sandbox = qs.Sandbox(
        template=qs.Templates.build(os.path.dirname(__file__) + "/", name="sat"),
        timeout=60,
        env={
            "CHUTES_API_KEY": CHUTES_API_KEY,
            "MODEL": MODEL,
            "SLUG": SLUG
        },
    )
    print('build docker')
    # Use elegant proxy interface (waits for /health and calls /run)
    try:
        result = sandbox.proxy.run(x=1, y=2, timeout=60)
        print(result)
    finally:
        sandbox.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
