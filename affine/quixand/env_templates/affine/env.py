#!/usr/bin/env python3

import os
import sys
from loguru import logger
import time
import asyncio
import httpx
from typing import Dict, Any, List, Optional
from functools import partial
import importlib
import openai

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ENV_NAME = os.environ.get("ENV_NAME")

class EvaluatorRequest(BaseModel):
    """Request model for evaluation endpoint"""
    model: str
    base_url: str = "https://llm.chutes.ai/v1"
    ids: Optional[List[int]] = None
    timeout: int = 1200
    temperature: float = 0.7

class EvaluatorResponse(BaseModel):
    """Response model for evaluation endpoint"""
    task_name: str
    total_score: float
    success_rate: float
    num_evaluated: int
    time_taken: float
    details: List[Dict[str, Any]]

class LocalResponse(BaseModel):
    """Local Response class to avoid importing affine module"""
    response: Optional[str]
    latency_seconds: float = 0.0
    attempts: int = 1
    model: str
    error: Optional[str] = None
    success: bool = True
    timestamp: Optional[float] = Field(default_factory=time.time)

async def llm_chat(
    base_url: str, 
    model: str, 
    prompt: str, 
    timeout_secs: float = 600.0,
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


app = FastAPI(title=f"Affine Environment Server - {ENV_NAME}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "environment": ENV_NAME}

async def validate_api_key(api_key: str, base_url: str) -> bool:
    """Validate the API key by making a test request to the API"""
    if not api_key:
        return False
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0
            )
            return 200 <= response.status_code < 300
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False

@app.post("/evaluator", response_model=EvaluatorResponse)
async def evaluate_model(request: EvaluatorRequest):
    """
    Evaluate a model on Affine tasks.
    
    This endpoint allows evaluating language models on various Affine tasks
    by providing model configuration and task parameters.
    """
    try:
        module = importlib.import_module(f"affine.envs.{ENV_NAME.lower()}")

        api_key = os.environ.get("CHUTES_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API key must be provided in request or CHUTES_API_KEY environment variable"
            )

        is_valid = await validate_api_key(api_key, request.base_url)
        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid API key for {request.base_url}"
            )
        logger.info(f"API key validated successfully for {request.base_url}")

        env_class = getattr(module, ENV_NAME.upper())

        env = env_class()
        logger.info(f"Created environment instance: {env_class.__name__}")
        
        # Generate challenge
        start_time = time.time()
        details = []
        total_score, ok = 0.0, 0.0
        ids = request.ids or [0]
        for idx in ids:
            try:
                challenge = await env.generate()
                logger.info(f"Generated challenge: {challenge.prompt[:100]}...")

                llm_response = await llm_chat(
                    base_url=request.base_url,
                    model=request.model,
                    prompt=challenge.prompt,
                    timeout_secs=request.timeout,
                    temperature=request.temperature
                )
                logger.info(f"llm response: {llm_response[:100]}")
                
                response = LocalResponse(
                    response=llm_response,
                    model=request.model,
                    latency_seconds=0.0,
                    attempts=1,
                    error=None,
                    success=True
                )
                
                # Evaluate the response
                evaluation = await env.evaluate(challenge, response)
                total_score += evaluation.score
                ok += 1.0 if evaluation.score > 0 else 0.0
                details.append({
                    "id":  int(idx),
                    "reward": evaluation.score,
                    "success": bool(evaluation.score > 0),
                    "experiences": {"challenge": challenge.prompt, "llm_response": llm_response}
                })
            except Exception as e:
                details.append({"id": int(idx), "reward": 0.0, "success": False, "error": str(e)})
        time_taken = time.time() - start_time
        n = len(details) or 1
        return EvaluatorResponse(
            task_name=ENV_NAME,
            total_score=total_score,
            success_rate=ok / n,  # For single evaluation
            num_evaluated=n,
            time_taken=time_taken,
            details=details
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {type(e).__name__}: {repr(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info(f"Affine environment server ready for: {ENV_NAME}")
    logger.info(f"Available endpoints: /health, /evaluator")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info(f"Shutting down Affine environment server for: {ENV_NAME}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)