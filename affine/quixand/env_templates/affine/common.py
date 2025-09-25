import re
import os
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

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


async def _llm_chat(
    base_url: str, 
    model: str, 
    prompt: str, 
    timeout_secs: float = 600.0,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7
) -> str:
    if not base_url.strip():
        raise HTTPException(status_code=400, detail="base_url cannot be empty")
    if not model.strip():
        raise HTTPException(status_code=400, detail="model cannot be empty")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt cannot be empty")
    if timeout_secs <= 0:
        raise HTTPException(status_code=400, detail="timeout_secs must be positive")

    try:
        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip('/'),
            api_key=os.getenv("CHUTES_API_KEY"),
            timeout=httpx.Timeout(timeout_secs),
            max_retries=0
        )
        async def _make_request():
            return await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
        
        response = await _make_request()
        
        if not response.choices:
            raise HTTPException(status_code=502, detail="Empty response from API")
            
        content = response.choices[0].message.content
        if content is None:
            raise HTTPException(status_code=502, detail="Generated content is null")
        
        return content.strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                details.append({"id": int(idx), "reward": score, "success": bool(score > 0), "experiences": {"prompt": prompt, "content": content}})
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

