#!/usr/bin/env python3

import os
import sys
import importlib
import logging
from pathlib import Path

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inject_health_endpoint(app: FastAPI):
    """Inject a health check endpoint into the existing FastAPI app."""

    for route in app.routes:
        if hasattr(route, 'path') and route.path == '/health':
            logger.info("Health endpoint already exists, skipping injection")
            return

    @app.get("/health")
    async def health_check():
        return "ok"

    logger.info("Health endpoint injected successfully")


def create_app():
    env_name = os.environ.get("ENV_NAME", "")
    if not env_name:
        logger.error("ENV_NAME environment variable is not set")
    
    logger.info(f"Loading {env_name} environment server")
    
    module_name = f"agentenv_{env_name}.server"
    
    agentgym_path = Path("/app/AgentGym")
    env_path = agentgym_path / f"agentenv-{env_name}"
    
    if str(env_path) not in sys.path:
        sys.path.insert(0, str(env_path))
    
    if str(agentgym_path) not in sys.path:
        sys.path.insert(0, str(agentgym_path))
    
    logger.info(f"Python path updated: {sys.path[:2]}")
    try:
        logger.info(f"Importing module: {module_name}")
        server_module = importlib.import_module(module_name)
        app = server_module.app
        logger.info(f"Successfully loaded {env_name} environment app")
        
        inject_health_endpoint(app)
        
        return app
    except Exception as e:
        logger.error(f"Unexpected error loading environment: {e}")
        import traceback
        traceback.print_exc()
        

app = create_app()

@app.on_event("startup")
async def startup_event():
    env_name = os.environ.get("ENV_NAME", "unknown")
    logger.info(f"Environment server ready for: {env_name}")