#!/usr/bin/env python3

import os
import traceback
import random
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING, Type
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from pydantic import BaseModel, Field, validator, ValidationError

import affinetes as af_env
from affine.setup import logger


# Global environment cache
_ENV_CACHE: Dict[str, Any] = {}
_ENV_LOCK = Lock()


# ========================= Configuration =========================


@dataclass
class SandboxConfig:
    """Sandbox configuration"""

    timeout: int = 3600
    proxy_timeout: int = 700
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

    temperature: float = 0.7
    timeout: int = 600
    max_round: int = 10

    def to_payload(self, miner: Optional["Miner"] = None, **kwargs) -> Dict[str, Any]:
        """Convert to evaluator payload with support for dynamic parameters
        
        Args:
            miner: Optional Miner instance (can be None if model/base_url provided in kwargs)
            **kwargs: Additional parameters to override defaults (model, base_url, temperature, timeout, ids, etc.)
        """
        payload = {
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        
        # Add miner-based defaults if miner is provided
        if miner is not None:
            payload["model"] = miner.model
            payload["base_url"] = f"https://{miner.slug}.chutes.ai/v1"

        # Allow kwargs to override any default values
        payload.update(kwargs)

        return payload


class EnvType(Enum):
    """Environment types"""

    AFFINE = "affine"
    AGENTGYM = "agentgym"


# ========================= Models =========================


class Evaluation(BaseModel):
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)


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
                    logger.info(f"Using remote hosts for deployment: {hosts} (total replicas: {replicas})")

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

    async def _evaluate_single_miner(
        self, miner: Optional["Miner"] = None, **eval_kwargs
    ) -> Evaluation:
        """
        Common evaluation logic for a single miner

        Args:
            miner: Optional Miner instance (can be None if model/base_url in eval_kwargs)
            **eval_kwargs: Dynamic parameters (model, base_url, task_type, ids, etc.)

        Returns:
            Evaluation result
        """

        # Build payload with all dynamic parameters
        payload = self.evaluator_config.to_payload(miner, **eval_kwargs)

        # Execute evaluation
        try:
            timeout = (
                self.sandbox_config.proxy_timeout
                if self.env_type == EnvType.AFFINE
                else self.sandbox_config.proxy_timeout + 600
            )

            # Call affinetes evaluate method directly
            result = await self._env.evaluate(_timeout=timeout, **payload)

            return self._parse_evaluation_result(result, miner, eval_kwargs)

        except asyncio.TimeoutError as e:
            logger.error(f"Evaluation timeout for {self.env_name}: {e}, score set 0")
            return self._create_error_evaluation(e, miner, eval_kwargs)
        except Exception as e:
            raise

    def _parse_evaluation_result(
        self,
        result: Dict[str, Any],
        miner: "Miner",
        payload_extra: Dict[str, Any] = None,
    ) -> Evaluation:
        """Parse evaluation result"""
        total_score = float(result.get("total_score", 0.0))
        success_rate = float(result.get("success_rate", 0.0))
        details = result.get("details", [{}])[0]

        extra = {
            "success": bool(success_rate > 0),
            "details": details,
        }

        if payload_extra:
            extra.update(payload_extra)

        return Evaluation(score=total_score, extra=extra)

    def _create_error_evaluation(
        self, error: Exception, miner: "Miner", payload_extra: Dict[str, Any] = None
    ) -> Evaluation:
        """Create error evaluation"""
        extra = {
            "success": False,
            "error": str(error),
            "miner": miner,
        }

        if payload_extra:
            extra.update(payload_extra)

        return Evaluation(score=0.0, extra=extra)

    async def _evaluate_miners_batch(
        self, miners: Union["Miner", Dict[str, "Miner"]], evaluate_func
    ) -> Union[Evaluation, Dict[str, Evaluation]]:
        """
        Common batch evaluation logic

        Args:
            miners: Single miner or dict of miners
            evaluate_func: Function to evaluate single miner

        Returns:
            Evaluation or dict of evaluations
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
        return hasattr(miner, "model") and hasattr(miner, "slug")

    @abstractmethod
    async def evaluate(self, miner: Union["Miner", Dict[str, Any]]) -> "Evaluation":
        """Evaluate a single miner"""
        pass

    async def evaluate_batch(
        self, miners: List[Union["Miner", Dict[str, Any]]]
    ) -> List["Evaluation"]:
        """Evaluate multiple miners in parallel"""
        tasks = [self.evaluate(m) for m in miners]
        return await asyncio.gather(*tasks)

    def generate_random_task_id(self) -> int:
        """Generate a random task ID for this environment"""
        data_len = getattr(self, "data_len", 1)
        return random.randint(0, data_len - 1) % data_len

    async def evaluate_and_build_result(self, miner: "Miner") -> "Result":
        """
        Evaluate a miner and build a complete Result object.
        This method encapsulates all complexity for runner usage.
        
        Args:
            miner: Miner instance to evaluate
            
        Returns:
            Result object with evaluation results
        """
        import time
        from affine.models import Result
        
        start = time.monotonic()
        
        try:
            evaluation_result = await self.evaluate(miner)

            return Result(
                miner=miner,
                env=self.env_name,
                score=evaluation_result.score,
                latency_seconds=time.monotonic() - start,
                success=evaluation_result.extra.get("success", False),
                error=None,
                extra=evaluation_result.extra,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for {self.env_name} uid={miner.uid}: {e}")
            traceback.print_exc()
            
            return Result(
                miner=miner,
                env=self.env_name,
                score=0.0,
                latency_seconds=time.monotonic() - start,
                success=False,
                error=str(e),
                extra={"evaluation_failed": True},
                timestamp=time.time()
            )


# ========================= Environment Implementations =========================


class AffineSDKEnv(BaseSDKEnv):
    """Base class for Affine environments (SAT, ABD, DED, HVM, ELR)"""

    @property
    def env_type(self) -> EnvType:
        return EnvType.AFFINE

    @property
    def docker_image(self) -> str:
        """All Affine environments use the same image"""
        return "bignickeye/affine:latest"

    @property
    def env_vars(self) -> Dict[str, str]:
        """Affine environment variables"""
        env_vars = super().env_vars
        # Extract env name from template (e.g., "affine:sat" -> "sat")
        env_name = self.env_name.split(":", 1)[1] if ":" in self.env_name else self.env_name
        env_vars["ENV_NAME"] = env_name
        return env_vars

    async def evaluate(
        self, miner: Optional[Union["Miner", Dict[str, Any]]] = None,
        **eval_kwargs
    ) -> Union["Evaluation", Dict[str, "Evaluation"]]:
        """Evaluate using Affine environment endpoint.
        
        Args:
            miner: Optional Miner instance or dict of miners (can be None if model/base_url in eval_kwargs)
            **eval_kwargs: Dynamic parameters (model, base_url, temperature, task_type, num_samples, etc.)
        """

        # Extract env name from template (e.g., "affine:sat" -> "sat")
        env_name = self.env_name.split(":", 1)[1] if ":" in self.env_name else self.env_name
        
        # Set default task_type and num_samples if not provided in eval_kwargs
        eval_kwargs.setdefault("task_type", env_name)
        eval_kwargs.setdefault("num_samples", 1)

        async def evaluate_single(m):
            return await self._evaluate_single_miner(m, **eval_kwargs)

        return await self._evaluate_miners_batch(miner, evaluate_single)


class AgentGymSDKEnv(BaseSDKEnv):
    """Base class for AgentGym environments"""

    # Default configuration for each environment - can be overridden in subclasses
    DEFAULT_DATA_LEN = 200
    DEFAULT_MAX_ROUND = 10
    DEFAULT_TIMEOUT = 1200

    def __init__(self, data_len: int = None, max_round: int = None):
        super().__init__()
        # Use environment-specific defaults if not provided
        self.data_len = data_len if data_len is not None else self.DEFAULT_DATA_LEN
        self.max_round = max_round if max_round is not None else self.DEFAULT_MAX_ROUND

        # Update evaluator config
        if self._evaluator_config is None:
            self._evaluator_config = EvaluatorConfig(
                temperature=0.7, timeout=self.DEFAULT_TIMEOUT, max_round=self.max_round
            )
        else:
            self._evaluator_config.max_round = self.max_round
            self._evaluator_config.timeout = self.DEFAULT_TIMEOUT

    @property
    def env_type(self) -> EnvType:
        return EnvType.AGENTGYM

    @property
    def docker_image(self) -> str:
        """AgentGym environments have different images per task"""
        # Extract env name from template (e.g., "agentgym:webshop" -> "webshop")
        env_name = self.env_name.split(":", 1)[1] if ":" in self.env_name else self.env_name
        return f"bignickeye/agentgym:{env_name}"

    @property
    def env_vars(self) -> Dict[str, str]:
        """AgentGym environment variables"""
        env_vars = super().env_vars
        # Add AgentGym-specific variables
        env_vars["TODO_KEY"] = os.getenv("AGENTGYM_TOOL_TODO_KEY", "")
        env_vars["MOVIE_KEY"] = os.getenv("AGENTGYM_TOOL_MOVIE_KEY", "")
        env_vars["SHEET_EMAIL"] = os.getenv("AGENTGYM_TOOL_SHEET_EMAIL", "")
        return env_vars

    def _normalize_task_ids(self, task_id: Union[int, List[int], None]) -> List[int]:
        """Normalize task IDs to list format"""
        if task_id is None:
            return [random.randint(0, self.data_len - 1)]
        elif isinstance(task_id, int):
            return [task_id]
        elif isinstance(task_id, list):
            return task_id if task_id else [random.randint(0, self.data_len - 1)]
        else:
            raise TypeError(
                f"task_id must be int, list[int], or None, got {type(task_id)}"
            )

    async def evaluate(
        self,
        miner: Optional[Union["Miner", Dict[str, Any]]] = None,
        **eval_kwargs
    ) -> Union["Evaluation", Dict[str, "Evaluation"]]:
        """Evaluate using AgentGym environment endpoint.
        
        Args:
            miner: Optional Miner instance or dict of miners (can be None if model/base_url in eval_kwargs)
            **eval_kwargs: Dynamic parameters (model, base_url, temperature, ids, max_round, etc.)
        """

        # Handle task_id/ids parameter - set default if not provided
        if "ids" not in eval_kwargs and "task_id" not in eval_kwargs:
            eval_kwargs["ids"] = [random.randint(0, self.data_len - 1)]
        elif "task_id" in eval_kwargs and "ids" not in eval_kwargs:
            # Convert task_id to ids if needed
            eval_kwargs["ids"] = self._normalize_task_ids(eval_kwargs.pop("task_id"))
        
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
        ENV_REGISTRY[env_name] = cls
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


@register_env(EnvType.AFFINE, "affine:hvm")
class HVM(AffineSDKEnv):
    """HVM environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "affine:hvm"


@register_env(EnvType.AFFINE, "affine:elr")
class ELR(AffineSDKEnv):
    """ELR environment for SDK"""
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "affine:elr"


# AgentGym Environments
@register_env(EnvType.AGENTGYM, "agentgym:alfworld")
class ALFWORLD(AgentGymSDKEnv):
    """ALFWORLD environment for SDK"""
    DEFAULT_DATA_LEN = 2500
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:alfworld"


@register_env(EnvType.AGENTGYM, "agentgym:webshop")
class WEBSHOP(AgentGymSDKEnv):
    """WEBSHOP environment for SDK"""
    DEFAULT_MAX_ROUND = 10
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:webshop"


@register_env(EnvType.AGENTGYM, "agentgym:babyai")
class BABYAI(AgentGymSDKEnv):
    """BABYAI environment for SDK"""
    DEFAULT_DATA_LEN = 4000
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:babyai"


@register_env(EnvType.AGENTGYM, "agentgym:sciworld")
class SCIWORLD(AgentGymSDKEnv):
    """SCIWORLD environment for SDK"""
    DEFAULT_DATA_LEN = 4639
    DEFAULT_REPLICAS = 1

    @property
    def env_name(self) -> str:
        return "agentgym:sciworld"


@register_env(EnvType.AGENTGYM, "agentgym:textcraft")
class TEXTCRAFT(AgentGymSDKEnv):
    """TEXTCRAFT environment for SDK"""
    DEFAULT_DATA_LEN = 582
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
HVM_factory = create_env_factory(HVM)
ELR_factory = create_env_factory(ELR)
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
