#!/usr/bin/env python3

import os
import time
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from affine.core.models import Result
from affine.core.setup import logger
import affinetes as af_env


# ========================= Global Cache =========================

_ENV_CACHE: Dict[str, Any] = {}
_ENV_LOCK = Lock()


# ========================= Configuration =========================

@dataclass
class EnvConfig:
    """Environment-specific configuration"""
    name: str
    docker_image: str
    env_type: str = "affine"
    env_vars: Dict[str, str] = field(default_factory=dict)
    mem_limit: str = "10g"
    volumes: Optional[Dict[str, Dict[str, str]]] = None
    eval_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.0,
        "timeout": 600,
    })
    proxy_timeout: int = 600


# ========================= Environment Configurations =========================

# Canonical environment configurations
_ENV_CONFIGS_CANONICAL = {
    "affine:ded-v2": EnvConfig(
        name="affine:ded-v2",
        docker_image="affinefoundation/affine-env:v4",
        env_vars={"UVICORN_WORKERS": "4"},
        eval_params={
            "task_type": "ded",
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "affine:abd-v2": EnvConfig(
        name="affine:abd-v2",
        docker_image="affinefoundation/affine-env:v4",
        env_vars={"UVICORN_WORKERS": "4"},
        eval_params={
            "task_type": "abd",
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    
    # PrimeIntellect environments (no task_type)
    "cde": EnvConfig(
        name="cde",
        docker_image="affinefoundation/cde:pi",
        mem_limit="20g",
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "lgc": EnvConfig(
        name="lgc",
        docker_image="affinefoundation/lgc:pi",
        env_vars={"UVICORN_WORKERS": "8"},
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "mth": EnvConfig(
        name="mth",
        docker_image="affinefoundation/mth:pi",
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "sci": EnvConfig(
        name="sci",
        docker_image="affinefoundation/sci:pi",
        eval_params={
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    
    # AgentGym environments (require max_round)
    "agentgym:alfworld": EnvConfig(
        name="agentgym:alfworld",
        docker_image="affinefoundation/agentgym:alfworld",
        env_type="agentgym",
        eval_params={
            "max_round": 30,
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "agentgym:webshop": EnvConfig(
        name="agentgym:webshop",
        docker_image="affinefoundation/agentgym:webshop",
        env_type="agentgym",
        eval_params={
            "max_round": 30,
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "agentgym:babyai": EnvConfig(
        name="agentgym:babyai",
        docker_image="affinefoundation/agentgym:babyai",
        env_type="agentgym",
        eval_params={
            "max_round": 30,
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "agentgym:sciworld": EnvConfig(
        name="agentgym:sciworld",
        docker_image="affinefoundation/agentgym:sciworld",
        env_type="agentgym",
        eval_params={
            "max_round": 30,
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    "agentgym:textcraft": EnvConfig(
        name="agentgym:textcraft",
        docker_image="affinefoundation/agentgym:textcraft",
        env_type="agentgym",
        eval_params={
            "max_round": 30,
            "temperature": 0.0,
            "timeout": 600,
        },
    ),
    
    # SWE-bench Pro environment (requires DOOD)
    "swe-pro": EnvConfig(
        name="swe-pro",
        docker_image="affinefoundation/swebench:pro",
        env_type="swebench",
        env_vars={"UVICORN_WORKERS": "3"},
        mem_limit="10g",
        volumes={
            "/var/run/docker.sock": {
                "bind": "/var/run/docker.sock",
                "mode": "rw"
            }
        },
        eval_params={
            "max_iterations": 50,
            "temperature": 0.0,
            "timeout": 1800,
        },
        proxy_timeout=1800,
    ),
}

# Alias mappings (multiple names can map to the same canonical config)
_ENV_ALIASES = {
    # ABD aliases - all point to v2
    "affine:abd": "affine:abd-v2",
    "abd": "affine:abd-v2",
    "abd-v2": "affine:abd-v2",
    
    # DED aliases - all point to v2
    "affine:ded": "affine:ded-v2",
    "ded": "affine:ded-v2",
    "ded-v2": "affine:ded-v2",
    
    # SAT aliases
    "sat": "affine:sat",
    
    # PrimeIntellect aliases (uppercase versions)
    "CDE": "cde",
    "LGC": "lgc",
    "MTH": "mth",
    "SCI": "sci",
    
    # SWE-bench aliases
    "SWE-PRO": "swe-pro",
}

# Build final ENV_CONFIGS with aliases
ENV_CONFIGS = {}
for canonical_name, config in _ENV_CONFIGS_CANONICAL.items():
    ENV_CONFIGS[canonical_name] = config

# Add all aliases
for alias, canonical in _ENV_ALIASES.items():
    if canonical in _ENV_CONFIGS_CANONICAL:
        ENV_CONFIGS[alias] = _ENV_CONFIGS_CANONICAL[canonical]


# ========================= Base Environment =========================

class SDKEnvironment:
    """Unified SDK environment implementation"""
    
    def __init__(self, env_name: str):
        if env_name not in ENV_CONFIGS:
            raise ValueError(f"Unknown environment: {env_name}")
        
        self.config = ENV_CONFIGS[env_name]
        self._env = self._load_environment()
        self._env_lock = asyncio.Lock()
    
    @property
    def env_name(self) -> str:
        return self.config.name
    
    @property
    def env_type(self) -> str:
        return self.config.env_type
    
    @property
    def docker_image(self) -> str:
        return self.config.docker_image
    
    def _get_env_vars(self) -> Dict[str, str]:
        """Get environment variables for this environment"""
        api_key = os.getenv("CHUTES_API_KEY")
        if not api_key:
            raise ValueError("CHUTES_API_KEY environment variable is required")
        
        env_vars = {"CHUTES_API_KEY": api_key}
        
        # Add ENV_NAME for affine environments (from task_type in eval_params)
        if "task_type" in self.config.eval_params:
            env_vars["ENV_NAME"] = self.config.eval_params["task_type"]
        
        env_vars.update(self.config.env_vars)
        return env_vars
    
    def _load_hosts_config(self) -> Dict[str, Any]:
        """Load hosts configuration from file"""
        # Check for config file in multiple locations
        config_paths = [
            Path(os.getenv("AFFINETES_HOSTS_CONFIG", "")),
            Path.cwd() / "affinetes_hosts.json",
            Path.home() / ".affine" / "hosts.json",
            Path("/etc/affine/hosts.json"),
        ]
        
        for config_path in config_paths:
            if config_path.exists() and config_path.is_file():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    logger.debug(f"Loaded hosts config from: {config_path}")
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load hosts config from {config_path}: {e}")
        
        return {}
    
    def _get_hosts_for_env(self) -> Optional[List[str]]:
        """Get hosts for this environment from config file or env var"""
        # Try config file first
        config = self._load_hosts_config()
        
        if config:
            # Check for environment-specific hosts
            if self.env_name in config:
                hosts = config[self.env_name]
                if isinstance(hosts, list) and hosts:
                    logger.debug(f"Using config file hosts for {self.env_name}: {hosts}")
                    return hosts
            
            # Fall back to default hosts in config
            if "default" in config:
                hosts = config["default"]
                if isinstance(hosts, list) and hosts:
                    logger.debug(f"Using default config hosts for {self.env_name}: {hosts}")
                    return hosts
        
        # Fall back to environment variable (for backward compatibility)
        hosts_env = os.getenv("AFFINETES_HOSTS", "").strip()
        if hosts_env:
            hosts = [h.strip() for h in hosts_env.split(",") if h.strip()]
            if hosts:
                logger.debug(f"Using env var hosts for {self.env_name}: {hosts}")
                return hosts
        
        return ["localhost"]
    
    def _load_environment(self) -> Any:
        """Load or get cached environment instance"""
        with _ENV_LOCK:
            if self.env_name in _ENV_CACHE:
                cached = _ENV_CACHE[self.env_name]
                if cached.is_ready():
                    logger.debug(f"Reusing cached environment: {self.env_name}")
                    return cached
                del _ENV_CACHE[self.env_name]
            
            # Get hosts for this environment
            hosts = self._get_hosts_for_env()
            
            # Load environment
            logger.info(f"Loading environment: {self.env_name} (image={self.docker_image}, hosts={hosts or 'local'}, mem_limit={self.config.mem_limit})")
            
            # Build load_env kwargs
            load_kwargs = {
                "image": self.docker_image,
                "mode": "docker",
                "replicas": len(hosts),
                "env_vars": self._get_env_vars(),
                "hosts": hosts,
                "container_name": self.env_name.replace(":", "-"),
                "mem_limit": self.config.mem_limit,
                "pull": True,
                "force_recreate": True,
            }
            
            # Add volumes if configured
            if self.config.volumes:
                load_kwargs["volumes"] = self.config.volumes
            
            env = af_env.load_env(**load_kwargs)
            
            _ENV_CACHE[self.env_name] = env
            logger.debug(f"Cached environment: {self.env_name}")
            return env
    
    def _generate_seed(self, task_id: int) -> int:
        """Generate deterministic seed"""
        seed_string = f"{self.env_name}:{task_id}"
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big') % (2**32)
    
    def _prepare_eval_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare evaluation kwargs based on environment configuration"""
        if "task_id" not in kwargs:
            raise ValueError("task_id is required for evaluation")
        
        # Generate seed if not provided
        if "seed" not in kwargs:
            kwargs["seed"] = self._generate_seed(kwargs["task_id"])
        
        # Merge eval_params from config (user-provided kwargs take precedence)
        for key, value in self.config.eval_params.items():
            kwargs.setdefault(key, value)
        
        return kwargs
    
    async def _evaluate_single(self, miner: Optional["Miner"], **kwargs) -> Result:
        """Evaluate single miner"""
        start = time.monotonic()
        kwargs = self._prepare_eval_kwargs(**kwargs)
        
        # Build payload with miner info
        payload = kwargs.copy()
        if miner and hasattr(miner, 'slug') and miner.slug:
            payload.update({
                "model": miner.model,
                "base_url": f"https://{miner.slug}.chutes.ai/v1"
            })
        
        result = await self._env.evaluate(_timeout=self.config.proxy_timeout, **payload)
        
        return self._build_result(result, miner, payload, start)
    
    def _build_result(self, result: Dict[str, Any], miner: Optional["Miner"], 
                     payload: Dict[str, Any], start_time: float) -> Result:
        """Build Result object from evaluation result"""
        extra = result.get("extra", {}).copy()
        extra["image"] = self.docker_image
        extra["request"] = payload.copy()
        
        return Result(
            miner_hotkey=miner.hotkey if miner else "",
            model_revision=miner.revision if miner else "",
            env=self.env_name,
            score=float(result.get("score", 0.0)),
            latency_seconds=time.monotonic() - start_time,
            success=bool(result.get("success", False)),
            error=result.get("error"),
            task_id=payload.get("task_id"),
            extra=extra,
            timestamp=time.time()
        )
    
    async def evaluate(self, miner: Optional[Union["Miner", Dict[str, "Miner"]]] = None, 
                      **kwargs) -> Union[Result, Dict[str, Result]]:
        """Evaluate miner(s)"""
        if isinstance(miner, dict):
            results = {}
            for key, m in miner.items():
                if self._validate_miner(m):
                    results[key] = await self._evaluate_single(m, **kwargs)
                else:
                    logger.warning(f"Skipping invalid miner: {key}")
            return results
        else:
            return await self._evaluate_single(miner, **kwargs)
    
    async def evaluate_batch(self, miners: List[Union["Miner", Dict[str, Any]]], 
                            **kwargs) -> List[Result]:
        """Evaluate multiple miners in parallel"""
        tasks = [self.evaluate(m, **kwargs) for m in miners]
        return await asyncio.gather(*tasks)
    
    @staticmethod
    def _validate_miner(miner: Any) -> bool:
        """Validate miner object"""
        return (hasattr(miner, "model") and hasattr(miner, "slug") and 
                miner.model and miner.slug)


# ========================= Factory Functions =========================

def create_environment(env_name: str) -> SDKEnvironment:
    """Create environment by name"""
    return SDKEnvironment(env_name)


def list_available_environments() -> Dict[str, List[str]]:
    """List all available environments grouped by type"""
    result = {}
    for name, config in ENV_CONFIGS.items():
        env_type = config.env_type
        result.setdefault(env_type, []).append(name)
    
    for env_type in result:
        result[env_type].sort()
    
    return result


def cleanup_all_environments():
    """Clean up all cached environments"""
    with _ENV_LOCK:
        logger.info("Cleaning up all cached environments")
        for name, env in list(_ENV_CACHE.items()):
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(env.cleanup())
                logger.debug(f"Cleaned up environment: {name}")
            except Exception as e:
                logger.warning(f"Error cleaning up environment {name}: {e}")
        
        _ENV_CACHE.clear()


# ========================= Backward Compatibility Aliases =========================

# Factory functions for backward compatibility
SAT_factory = lambda: create_environment("sat")
ABD_factory = lambda: create_environment("abd")  # Points to abd-v2
DED_factory = lambda: create_environment("ded")  # Points to ded-v2
DED_V2_factory = lambda: create_environment("ded-v2")
ABD_V2_factory = lambda: create_environment("abd-v2")
CDE_factory = lambda: create_environment("cde")
LGC_factory = lambda: create_environment("lgc")
MTH_factory = lambda: create_environment("mth")
SCI_factory = lambda: create_environment("sci")
ALFWORLD_factory = lambda: create_environment("agentgym:alfworld")
WEBSHOP_factory = lambda: create_environment("agentgym:webshop")
BABYAI_factory = lambda: create_environment("agentgym:babyai")
SCIWORLD_factory = lambda: create_environment("agentgym:sciworld")
TEXTCRAFT_factory = lambda: create_environment("agentgym:textcraft")
SWE_PRO_factory = lambda: create_environment("swe-pro")

# Legacy class aliases
SAT = SAT_factory
ABD = ABD_factory
DED = DED_factory
DED_V2 = DED_V2_factory
ABD_V2 = ABD_V2_factory
CDE = CDE_factory
LGC = LGC_factory
MTH = MTH_factory
SCI = SCI_factory
ALFWORLD = ALFWORLD_factory
WEBSHOP = WEBSHOP_factory
BABYAI = BABYAI_factory
SCIWORLD = SCIWORLD_factory
TEXTCRAFT = TEXTCRAFT_factory

# SWE-bench factories
SWE_PRO = SWE_PRO_factory
