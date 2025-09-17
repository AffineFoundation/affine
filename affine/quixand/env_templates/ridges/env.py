import os
import json
import logging
import traceback
import asyncio
import time
from typing import Optional
from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not os.getenv("CHUTES_API_KEY") and os.getenv("CHUTES_API_KEY"):
    logger.warning("CHUTES_API_KEY is not set. Proxy endpoints may not work correctly.")
    exit(0)

class AgentRequest(BaseModel):
    problem_statement: str = Field(description="The problem statement to solve")
    run_id: Optional[str] = Field(default="default", description="Run ID for tracking")

class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="Text to embed")
    run_id: str = Field(..., description="Evaluation run ID")

class GPTMessage(BaseModel):
    role: str
    content: str

class InferenceRequest(BaseModel):
    run_id: str = Field(..., description="Evaluation run ID")
    model: Optional[str] = Field(None, description="Model to use for inference")
    temperature: Optional[float] = Field(None, description="Temperature for inference")
    messages: List[GPTMessage] = Field(..., description="Messages to send to the model")

app = FastAPI(
    title="Ridges Agent API",
    description="API wrapper for Ridges AI",
)

@lru_cache(maxsize=1)
def get_chutes_client():
    from proxy.chutes_client import ChutesClient
    return ChutesClient()

@lru_cache(maxsize=1)
def get_inference_manager():
    from proxy.providers import InferenceManager
    return InferenceManager()


@app.post("/agents/embedding")
async def embedding_endpoint(request: EmbeddingRequest):
    try:
        logger.info(f"Embedding request input {request.input}")
        chutes_client = get_chutes_client()
        embedding_result = await chutes_client.embed(None, request.input)
        logger.info(f"Embedding request completed successfully")
        return embedding_result
    except HTTPException:
        raise
    except Exception as e:
        # More detailed error logging for debugging
        logger.error(f"Embedding request for run_id {request.run_id} -- error: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail="Failed to get embedding due to internal server error. Please try again later."
        )

@app.post("/agents/inference")
async def inference_endpoint(request: InferenceRequest):
    try:
        temperature = request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
        model = request.model if request.model is not None else DEFAULT_MODEL
        inference_manager = get_inference_manager()
        inference_result = await inference_manager.inference(
            run_id=None,
            messages=request.messages,
            temperature=temperature,
            model=model
        )
        try:
            if isinstance(inference_result, str):
                resp_preview = (inference_result[:200] + "…") if len(inference_result) > 200 else inference_result
            else:
                resp_preview = str(inference_result)[:200]
        except Exception:
            resp_preview = "<non-string response>"
        logger.info("Inference request completed successfully")
        return inference_result
    except HTTPException:
        logger.error(f"HTTPException in inference endpoint")
        raise
    except Exception as e:
        # More detailed error logging for debugging
        import traceback
        logger.error(f"Inference request for run_id {request.run_id} (model: {request.model}) -- error: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail="Failed to get inference due to internal server error. Please try again later."
        )

@app.get("/health")
async def health_check():
    return "OK"

@app.post("/run")
async def solve_problem(request: AgentRequest):
    if not request.problem_statement:
        logger.warning("No problem statement provided in request")
        return {
            "success": False,
            "error": "No problem statement provided. Please provide 'problem_statement' in request body"
        }

    try:
        problem_statement = request.problem_statement
        agent_input = {
            "problem_statement": problem_statement,
        }
        with open("input.json", "w", encoding="utf-8") as f:
            json.dump(agent_input, f)
        logger.info(f"Processing problem: {problem_statement[:100]}...")
        loop = asyncio.get_running_loop()
    
        try:
            from api.top.rank_1.agent import agent_main
            result = await loop.run_in_executor(None, agent_main, agent_input)
            output = {"success": True, "output": result}
            logger.info("Agent execution completed successfully")
        except Exception as e:
            logger.error(f"Exception during agent execution: {str(e)}", exc_info=True)
            error = traceback.format_exc()
            output = {"success": False, "error": error}
    except Exception as e:
        logger.error(f"Exception during agent execution: {str(e)}", exc_info=True)
        error = traceback.format_exc()
        output = {'success': False, 'error': error}

    with open('output.json', 'w') as f:
        json.dump(output, f)

    return output

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

def main():
    import quixand as qs
    chutes_api_key = os.getenv("CHUTES_API_KEY")

    print("=== Ridges Agent Example ===\n")

    image = qs.Templates.build("env_templates/ridges", name="ridges-agent")
    print(f"Image built: {image}\n")
    
    sandbox = qs.Sandbox(
        template=image,
        timeout=300,
        env={
            "CHUTES_API_KEY": chutes_api_key,
            "AGENT_MODEL": "deepseek-ai/DeepSeek-V3",
        },
    )
    print(f"Container ID: {sandbox.container_id[:8]}\n")
    
    try:
        problem = "Write a quicksort algorithm in Python without comments"
        print(f"Problem: {problem}")
        response = sandbox.proxy.run(
            port=8000,
            path="/run",
            method="POST",
            payload={
                "problem_statement": problem,
            },
            timeout=300
        )
        print(f"Agent response: {response}")
    finally:
        print("\nShutting down container...")
        sandbox.shutdown()
        print("Done")


if __name__ == "__main__":
    main()