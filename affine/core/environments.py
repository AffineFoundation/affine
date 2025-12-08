#!/usr/bin/env python3

import os
import time
import random
import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Type, Tuple
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from affine.core.models import Result
from affine.core.setup import logger
import affinetes as af_env


# Global environment cache
_ENV_CACHE: Dict[str, Any] = {}
_ENV_LOCK = Lock()


# ========================= Configuration =========================


@dataclass
class SandboxConfig:
    """Sandbox configuration"""

    timeout: int = 1200
    proxy_timeout: int = 600
    env: Dict[str, str] = None

    def __post_init__(self):
        if self.env is None:
            self.env = {
                "NO_PROXY": "localhost,127.0.0.1",
                "PYTHONPATH": "/app",
            }


@dataclass
class EvaluatorConfig:
    """Evaluator configuration"""

    temperature: float = 0.0
    timeout: int = 600

    def to_payload(self, miner: Optional["Miner"] = None, **kwargs) -> Dict[str, Any]:
        """Convert to evaluator payload with support for dynamic parameters
        
        Args:
            miner: Optional Miner instance (can be None if model/base_url provided in kwargs)
            **kwargs: Additional parameters to override defaults (model, base_url, temperature, timeout, task_id, etc.)
        """
        payload = {
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        
        # Add miner-based defaults if miner is provided and has valid slug
        if miner is not None and hasattr(miner, 'slug') and miner.slug is not None:
            payload["model"] = miner.model
            payload["base_url"] = f"https://{miner.slug}.chutes.ai/v1"

        # Allow kwargs to override any default values
        payload.update(kwargs)

        return payload


class EnvType(Enum):
    """Environment types"""

    AFFINE = "affine"
    AGENTGYM = "agentgym"


# ========================= Base Classes =========================


class BaseSDKEnv(ABC):
    """Base class for all SDK environments"""

    # Class-level configuration
    _sandbox_config: SandboxConfig = None
    _evaluator_config: EvaluatorConfig = None
    DEFAULT_REPLICAS: int = 1

    def __init__(self):
        super().__init__()
        self._env = self._load_environment()
        self._env_lock = asyncio.Lock()

    @property
    def sandbox_config(self) -> SandboxConfig:
        """Get sandbox configuration"""
        if self._sandbox_config is None:
            self._sandbox_config = SandboxConfig()
        return self._sandbox_config

    @property
    def evaluator_config(self) -> EvaluatorConfig:
        """Get evaluator configuration"""
        if self._evaluator_config is None:
            self._evaluator_config = EvaluatorConfig()
        return self._evaluator_config

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Return environment name"""
        pass

    @property
    @abstractmethod
    def env_type(self) -> EnvType:
        """Return environment type"""
        pass

    @property
    def docker_image(self) -> str:
        """Return Docker image for this environment"""
        raise NotImplementedError("Subclass must implement docker_image property")

    @property
    def env_vars(self) -> Dict[str, str]:
        """Return environment variables for this environment"""
        api_key = os.getenv("CHUTES_API_KEY")
        if not api_key:
            raise ValueError("CHUTES_API_KEY environment variable is required")
        return {"CHUTES_API_KEY": api_key}

    def _load_environment(self) -> Any:
        """Load or get cached environment instance"""
        with _ENV_LOCK:
            template = self.env_name
            
            # Check cache for shared instances
            if template in _ENV_CACHE:
                cached_env = _ENV_CACHE[template]
                if cached_env.is_ready():
                    logger.debug(f"Reusing cached environment: {template}")
                    return cached_env
                else:
                    logger.debug(f"Removing stale cached environment: {template}")
                    del _ENV_CACHE[template]
            
            # Parse AFFINETES_HOSTS environment variable
            hosts_env = os.getenv("AFFINETES_HOSTS", "").strip()
            hosts = None
            replicas = self.DEFAULT_REPLICAS
            
            if hosts_env:
                # Parse comma-separated hosts
                parsed_hosts = [h.strip() for h in hosts_env.split(",") if h.strip()]
                if parsed_hosts:
                    hosts = parsed_hosts
                    # When using remote hosts, total replicas = DEFAULT_REPLICAS * number of hosts
                    replicas = self.DEFAULT_REPLICAS * len(hosts)
                    logger.debug(f"Using remote hosts for deployment: {hosts} (total replicas: {replicas})")

            # Generate container name based on environment name
            container_name = template.replace(":", "-")
            
            # Load environment using affinetes
            logger.info(f"Loading environment: {template} (image={self.docker_image}, replicas={replicas})")
            environment = af_env.load_env(
                image=self.docker_image,
                mode="docker",
                env_vars=self.env_vars,
                hosts=hosts,
                replicas=replicas,
                container_name=container_name,
                mem_limit="10g",
                pull=True,
                force_recreate=True,
            )
            
            # Cache the environment
            _ENV_CACHE[template] = environment
            logger.debug(f"Cached environment: {template}")
            
            return environment

    def _generate_deterministic_seed(self, task_id: int) -> int:
        """Generate deterministic seed based on env_name and task_id.
        
        Args:
            task_id: The task ID
            
        Returns:
            A deterministic seed value
        """
        # Combine env_name and task_id to create a unique string
        seed_string = f"{self.env_name}:{task_id}"
        # Use SHA256 hash to generate deterministic seed
        hash_object = hashlib.sha256(seed_string.encode())
        # Convert first 8 bytes of hash to integer and modulo to fit in 32-bit range
        seed = int.from_bytes(hash_object.digest()[:8], byteorder='big') % (2**32)
        return seed

    async def _evaluate_single_miner(
        self, miner: Optional["Miner"] = None, **eval_kwargs
    ) -> "Result":
        """
        Common evaluation logic for a single miner

        Args:
            miner: Optional Miner instance (can be None if model/base_url in eval_kwargs)
            **eval_kwargs: Dynamic parameters (model, base_url, task_type, task_id, etc.)

        Returns:
            Result object with evaluation results
        """
        start = time.monotonic()

        # Generate deterministic seed based on env_name and task_id if not provided
        if 'seed' not in eval_kwargs:
            task_id = eval_kwargs.get('task_id')
            if task_id is not None:
                eval_kwargs['seed'] = self._generate_deterministic_seed(task_id)
            else:
                # Fallback to random if task_id is not available
                eval_kwargs['seed'] = random.randint(0, 2**32 - 1)
        
        # Build payload with all dynamic parameters (including seed)
        payload = self.evaluator_config.to_payload(miner, **eval_kwargs)

        # Call affinetes evaluate method directly
        result = await self._env.evaluate(_timeout=self.sandbox_config.proxy_timeout, **payload)

        return self._parse_evaluation_result(result, miner, payload, start)

    def _parse_evaluation_result(
        self,
        result: Dict[str, Any],
        miner: Optional["Miner"],
        payload_extra: Dict[str, Any] = None,
        start_time: float = None,
    ) -> "Result":
        """Parse evaluation result and construct Result"""
        
        # Extract top-level fields
        score = float(result.get("score", 0.0))
        success = bool(result.get("success", False))
        error = result.get("error")
        extra = result.get("extra", {}).copy()
        
        extra['image'] = self.docker_image
        if payload_extra:
            extra['request'] = payload_extra.copy()
        
        # Extract task_id from payload if available (for sequential sampling tracking)
        task_id = None
        if payload_extra and 'task_id' in payload_extra:
            task_id = payload_extra['task_id']
        
        # Extract miner info (hotkey + revision)
        miner_hotkey = ""
        model_revision = ""
        if miner:
            miner_hotkey = miner.hotkey
            model_revision = miner.revision or ""

        return Result(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=self.env_name,
            score=score,
            latency_seconds=time.monotonic() - start_time if start_time else 0.0,
            success=success,
            error=error,
            task_id=task_id,
            extra=extra,
            timestamp=time.time()
        )

    def _create_error_result(
        self, error: Exception, miner: Optional["Miner"], payload_extra: Dict[str, Any] = None, start_time: float = None
    ) -> "Result":
        extra = {
            "image": self.docker_image,
            "request": payload_extra,
        }
        
        # Extract task_id from payload if available
        task_id = None
        if payload_extra and 'task_id' in payload_extra:
            task_id = payload_extra['task_id']
        
        # Extract miner info (hotkey + revision)
        miner_hotkey = ""
        model_revision = ""
        if miner:
            miner_hotkey = miner.hotkey
            model_revision = miner.revision or ""

        return Result(
            miner_hotkey=miner_hotkey,
            model_revision=model_revision,
            env=self.env_name,
            score=0.0,
            latency_seconds=time.monotonic() - start_time if start_time else 0.0,
            success=False,
            error=str(error),
            task_id=task_id,
            extra=extra,
            timestamp=time.time()
        )

    async def _evaluate_miners_batch(
        self, miners: Union["Miner", Dict[str, "Miner"]], evaluate_func
    ) -> Union["Result", Dict[str, "Result"]]:
        """
        Common batch evaluation logic

        Args:
            miners: Single miner or dict of miners
            evaluate_func: Function to evaluate single miner

        Returns:
            Result or dict of results
        """
        if isinstance(miners, dict):
            results = {}
            for key, miner in miners.items():
                if not self._validate_miner(miner):
                    logger.warning(f"Skipping invalid miner entry: {key}")
                    continue
                results[key] = await evaluate_func(miner)
            return results
        else:
            return await evaluate_func(miners)

    def _validate_miner(self, miner: Any) -> bool:
        """Validate miner object"""
        return (
            hasattr(miner, "model")
            and hasattr(miner, "slug")
            and miner.model is not None
            and miner.slug is not None
        )

    @abstractmethod
    async def evaluate(self, miner: Union["Miner", Dict[str, Any]]) -> "Result":
        """Evaluate a single miner"""
        pass

    async def evaluate_batch(
        self, miners: List[Union["Miner", Dict[str, Any]]]
    ) -> List["Result"]:
        """Evaluate multiple miners in parallel"""
        tasks = [self.evaluate(m) for m in miners]
        return await asyncio.gather(*tasks)



# ========================= Environment Implementations =========================


class AffineSDKEnv(BaseSDKEnv):
    """Base class for Affine environments (SAT, ABD, DED, CDE, LGC, MTH, SCI)"""

    # Default Docker image for Affine environments
    DOCKER_IMAGE = "bignickeye/affine:v3"

    def __init__(self):
        super().__init__()

    @property
    def env_type(self) -> EnvType:
        return EnvType.AFFINE

    @property
    def docker_image(self) -> str:
        """All Affine environments use the same image"""
        return self.DOCKER_IMAGE

    def _get_base_task_name(self) -> str:
        """Extract and normalize task name, removing version suffixes.
        
        Examples:
            "affine:sat" -> "sat"
            "affine:ded-v2" -> "ded"
            "affine:abd-v2" -> "abd"
        """
        # Extract env name from template (e.g., "affine:sat" -> "sat")
        env_name = self.env_name.split(":", 1)[1] if ":" in self.env_name else self.env_name
        # Remove version suffix if present (e.g., "ded-v2" -> "ded")
        if "-v" in env_name:
            env_name = env_name.rsplit("-v", 1)[0]
        return env_name

    @property
    def env_vars(self) -> Dict[str, str]:
        """Affine environment variables"""
        env_vars = super().env_vars
        # Use base task name (without version suffix)
        env_vars["ENV_NAME"] = self._get_base_task_name()
        return env_vars

    async def evaluate(
        self, miner: Optional[Union["Miner", Dict[str, Any]]] = None,
        **eval_kwargs
    ) -> Union["Result", Dict[str, "Result"]]:
        """Evaluate using Affine environment endpoint.
        
        Args:
            miner: Optional Miner instance or dict of miners (can be None if model/base_url in eval_kwargs)
            **eval_kwargs: Dynamic parameters (model, base_url, temperature, task_type, task_id, etc.)
        """

        # Use base task name (without version suffix)
        base_task_name = self._get_base_task_name()
        
        # Set default task_type if not provided in eval_kwargs
        eval_kwargs.setdefault("task_type", base_task_name)
        
        # task_id must be provided by caller (from API server task queue)
        if "task_id" not in eval_kwargs:
            raise ValueError("task_id is required for evaluation")

        async def evaluate_single(m):
            return await self._evaluate_single_miner(m, **eval_kwargs)

        return await self._evaluate_miners_batch(miner, evaluate_single)


class AgentGymSDKEnv(BaseSDKEnv):
    """Base class for AgentGym environments"""

    DEFAULT_MAX_ROUND = 30
    DEFAULT_TIMEOUT = 600

    def __init__(self, max_round: int = None):
        super().__init__()
        self.max_round = max_round if max_round is not None else self.DEFAULT_MAX_ROUND
        self.evaluator_config.max_round = self.max_round
        self.evaluator_config.timeout = self.DEFAULT_TIMEOUT

    @property
    def env_type(self) -> EnvType:
        return EnvType.AGENTGYM

    @property
    def docker_image(self) -> str:
        """AgentGym environments have different images per task"""
        # Extract env name from template (e.g., "agentgym:webshop" -> "webshop")
        env_name = self.env_name.split(":", 1)[1] if ":" in self.env_name else self.env_name
        return f"affinefoundation/agentgym:{env_name}"

    @property
    def env_vars(self) -> Dict[str, str]:
        """AgentGym environment variables"""
        env_vars = super().env_vars
        # Add AgentGym-specific variables
        env_vars["TODO_KEY"] = os.getenv("AGENTGYM_TOOL_TODO_KEY", "")
        env_vars["MOVIE_KEY"] = os.getenv("AGENTGYM_TOOL_MOVIE_KEY", "")
        env_vars["SHEET_EMAIL"] = os.getenv("AGENTGYM_TOOL_SHEET_EMAIL", "")
        return env_vars

    async def evaluate(
        self,
        miner: Optional[Union["Miner", Dict[str, Any]]] = None,
        **eval_kwargs
    ) -> Union["Result", Dict[str, "Result"]]:
        """Evaluate using AgentGym environment endpoint.
        
        Args:
            miner: Optional Miner instance or dict of miners (can be None if model/base_url in eval_kwargs)
            **eval_kwargs: Dynamic parameters (model, base_url, temperature, task_id, max_round, etc.)
        """

        # task_id must be provided by caller (from API server task queue)
        if "task_id" not in eval_kwargs:
            raise ValueError("task_id is required for evaluation")
        
        # Set default max_round if not provided
        eval_kwargs.setdefault("max_round", self.max_round)

        async def evaluate_single(m):
            return await self._evaluate_single_miner(m, **eval_kwargs)

        return await self._evaluate_miners_batch(miner, evaluate_single)


# ========================= Concrete Environments =========================

# Environment registry for dynamic creation
ENV_REGISTRY = {}


def register_env(env_type: EnvType, env_name: str):
    """Decorator to register environment classes"""

    def decorator(cls):
        # Store with lowercase key for case-insensitive lookup
        ENV_REGISTRY[env_name.lower()] = cls
        cls._env_type = env_type
        cls._env_name = env_name
        return cls

    return decorator


# Affine Environments
@register_env(EnvType.AFFINE, "affine:sat")
class SAT(AffineSDKEnv):
    """SAT environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "affine:sat"


@register_env(EnvType.AFFINE, "affine:abd")
class ABD(AffineSDKEnv):
    """ABD environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "affine:abd"


@register_env(EnvType.AFFINE, "affine:ded")
class DED(AffineSDKEnv):
    """DED environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "affine:ded"


@register_env(EnvType.AFFINE, "affine:ded-v2")
class DED_V2(AffineSDKEnv):
    """DED-V2 environment for SDK"""
    DEFAULT_REPLICAS = 1
    DOCKER_IMAGE = "affinefoundation/affine-env:v4"

    @property
    def env_name(self) -> str:
        return "affine:ded-v2"


@register_env(EnvType.AFFINE, "affine:abd-v2")
class ABD_V2(AffineSDKEnv):
    """ABD-V2 environment for SDK"""
    DEFAULT_REPLICAS = 1
    DOCKER_IMAGE = "affinefoundation/affine-env:v4"

    @property
    def env_name(self) -> str:
        return "affine:abd-v2"


@register_env(EnvType.AFFINE, "CDE")
class CDE(AffineSDKEnv):
    """CDE environment for SDK"""
    DEFAULT_REPLICAS = 1
    DOCKER_IMAGE = "affinefoundation/cde:pi"

    @property
    def env_name(self) -> str:
        return "CDE"


@register_env(EnvType.AFFINE, "LGC")
class LGC(AffineSDKEnv):
    """LGC environment for SDK"""
    DEFAULT_REPLICAS = 1
    DOCKER_IMAGE = "affinefoundation/lgc:pi"

    @property
    def env_name(self) -> str:
        return "LGC"


@register_env(EnvType.AFFINE, "MTH")
class MTH(AffineSDKEnv):
    """MTH environment for SDK"""
    DEFAULT_REPLICAS = 1
    DOCKER_IMAGE = "affinefoundation/mth:pi"

    @property
    def env_name(self) -> str:
        return "MTH"


@register_env(EnvType.AFFINE, "SCI")
class SCI(AffineSDKEnv):
    """SCI environment for SDK"""
    DEFAULT_REPLICAS = 1
    DOCKER_IMAGE = "affinefoundation/sci:pi"

    @property
    def env_name(self) -> str:
        return "SCI"


# AgentGym Environments
@register_env(EnvType.AGENTGYM, "agentgym:alfworld")
class ALFWORLD(AgentGymSDKEnv):
    """ALFWORLD environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:alfworld"


@register_env(EnvType.AGENTGYM, "agentgym:webshop")
class WEBSHOP(AgentGymSDKEnv):
    """WEBSHOP environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:webshop"


@register_env(EnvType.AGENTGYM, "agentgym:babyai")
class BABYAI(AgentGymSDKEnv):
    """BABYAI environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:babyai"


@register_env(EnvType.AGENTGYM, "agentgym:sciworld")
class SCIWORLD(AgentGymSDKEnv):
    """SCIWORLD environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:sciworld"


@register_env(EnvType.AGENTGYM, "agentgym:textcraft")
class TEXTCRAFT(AgentGymSDKEnv):
    """TEXTCRAFT environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:textcraft"


# ========================= Factory Functions =========================


def create_env_factory(env_class: Type[BaseSDKEnv], **default_kwargs):
    """Create a factory function for environment"""

    def factory(**kwargs):
        merged_kwargs = {**default_kwargs, **kwargs}
        return env_class(**merged_kwargs)

    factory.__name__ = f"{env_class.__name__}_factory"
    factory.__doc__ = f"Create {env_class.__name__} environment"
    return factory


# Generate factory functions dynamically
SAT_factory = create_env_factory(SAT)
ABD_factory = create_env_factory(ABD)
DED_factory = create_env_factory(DED)
DED_V2_factory = create_env_factory(DED_V2)
ABD_V2_factory = create_env_factory(ABD_V2)
CDE_factory = create_env_factory(CDE)
LGC_factory = create_env_factory(LGC)
MTH_factory = create_env_factory(MTH)
SCI_factory = create_env_factory(SCI)
ALFWORLD_factory = create_env_factory(ALFWORLD)
WEBSHOP_factory = create_env_factory(WEBSHOP)
BABYAI_factory = create_env_factory(BABYAI)
SCIWORLD_factory = create_env_factory(SCIWORLD)
TEXTCRAFT_factory = create_env_factory(TEXTCRAFT)


# ========================= Utility Functions =========================


async def create_environment(env_name: str, **kwargs) -> BaseSDKEnv:
    """
    Create environment by name

    Args:
        env_name: Environment name
        **kwargs: Environment-specific parameters

    Returns:
        Environment instance

    Raises:
        ValueError: If environment name is unknown
    """
    env_class = ENV_REGISTRY.get(env_name.lower())
    if not env_class:
        raise ValueError(f"Unknown environment: {env_name}")

    return env_class(**kwargs)


def list_available_environments() -> Dict[str, List[str]]:
    """List all available environments grouped by type"""
    result = {}
    for env_name, env_class in ENV_REGISTRY.items():
        env_type = env_class._env_type.value
        if env_type not in result:
            result[env_type] = []
        result[env_type].append(env_name)

    for env_type in result:
        result[env_type].sort()

    return result


def cleanup_all_environments():
    """Clean up all cached environments"""
    with _ENV_LOCK:
        logger.info("Cleaning up all cached environments")
        for template, env in list(_ENV_CACHE.items()):
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(env.cleanup())
                logger.debug(f"Cleaned up environment: {template}")
            except Exception as e:
                logger.warning(f"Error cleaning up environment {template}: {e}")
        
        _ENV_CACHE.clear()
