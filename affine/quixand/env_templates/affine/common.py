import re
import os
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class EvaluatorRequest(BaseModel):
    model: str
    base_url: str = "https://llm.chutes.ai/v1"
    ids: Optional[List[int]] = None
    max_round: int = 10
    data_len: int = 200
    timeout: int = 1200


class EvaluatorResponse(BaseModel):
    task_name: str
    total_score: float
    success_rate: float
    num_evaluated: int
    time_taken: float
    details: List[Dict[str, Any]]


from utils import EvaluateRequest, EvaluateResponse, CreateResponse, normalize, strip_fences, run_program


 

# --------------------------- Injector (AgentGym‑style) --------------------------- #
def inject_health_endpoint(app: FastAPI):
    for route in app.routes:
        if getattr(route, 'path', None) == '/health':
            return
    @app.get("/health")
    async def health_check():
        return "ok"


async def _llm_chat(base_url: str, model: str, prompt: str) -> str:
    api_key = os.environ.get("CHUTES_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="CHUTES_API_KEY is not set")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 1.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        txt = await r.aread()
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=f"LLM error: {txt[:200].decode(errors='ignore')}")
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def inject_evaluator_endpoint(app: FastAPI, env_name: str):
    for route in app.routes:
        if getattr(route, 'path', None) == '/evaluator':
            return

    # Lazy import env module (abd|ded|sat) to get handlers
    import importlib
    mod = importlib.import_module(env_name)

    @app.post("/evaluator", response_model=EvaluatorResponse)
    async def evaluator_endpoint(request: EvaluatorRequest):
        start = time.time()
        ids = request.ids or [0]
        total, ok = 0.0, 0.0
        details: List[Dict[str, Any]] = []

        # Acquire handlers and models from env module
        CreateRequest = getattr(mod, 'CreateRequest', None)
        create_fn = getattr(mod, 'create_endpoint', None)
        EvaluateRequestModel = getattr(mod, 'EvaluateRequest', None)
        evaluate_fn = getattr(mod, 'evaluate_endpoint', None)
        local_strip = getattr(mod, 'strip_fences', None)

        if not (CreateRequest and create_fn and EvaluateRequestModel and evaluate_fn):
            raise HTTPException(status_code=500, detail=f"Env {env_name} missing required endpoints or schemas")

        for idx in ids:
            try:
                # Build challenge via /create (env-specific). Many envs support default CreateRequest().
                cr = CreateRequest()
                chal = await create_fn(cr)
                prompt = chal.prompt
                extra = chal.extra or {}

                # Query LLM once per id
                content = await _llm_chat(
                    base_url=request.base_url,
                    model=request.model,
                    prompt=prompt,
                )
                code = (local_strip or strip_fences)(content)
                ev = await evaluate_fn(EvaluateRequestModel(prompt=prompt, extra=extra, response=code))
                score = float(getattr(ev, 'score', 0.0))
                total += score
                ok += 1.0 if score > 0 else 0.0
                details.append({"id": int(idx), "reward": score, "success": bool(score > 0)})
            except Exception as e:
                details.append({"id": int(idx), "reward": 0.0, "success": False, "error": str(e)})

        n = len(details) or 1
        return EvaluatorResponse(
            task_name=env_name,
            total_score=total / n,
            success_rate=ok / n,
            num_evaluated=n,
            time_taken=time.time() - start,
            details=details,
        )

