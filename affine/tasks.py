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
from pydantic import BaseModel, Field, validator, ValidationError
from concurrent.futures import ThreadPoolExecutor

from affine.quixand.core.sandbox_manager import get_sandbox
from affine.setup import logger

# Global thread pool for sandbox blocking calls
_EXECUTOR = None

def get_executor() -> ThreadPoolExecutor:
    """Get or create global thread pool executor."""
    global _EXECUTOR
    if _EXECUTOR is None:
        max_workers = int(os.getenv("AFFINE_MAX_CONCURRENCY", "20")) * 2
        _EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sandbox_")
    return _EXECUTOR


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

    def to_payload(self, miner: "Miner", task_ids: List[int] = None) -> Dict[str, Any]:
        """Convert to evaluator payload"""
        payload = {
            "model": miner.model,
            "base_url": f"https://{miner.slug}.chutes.ai/v1",
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

        if task_ids is not None:
            payload["ids"] = task_ids

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

    def __init__(self):
        super().__init__()
        self._sandbox = self.get_sandbox()
        self._sandbox_lock = asyncio.Lock()

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

    def get_sandbox(self) -> Any:
        """Get or create sandbox instance"""
        return get_sandbox(
            template=self.env_name,
            shared=True,
            timeout=self.sandbox_config.timeout,
            env=self.sandbox_config.env,
        )

    async def _evaluate_single_miner(
        self, miner: "Miner", payload_extra: Dict[str, Any] = None
    ) -> Evaluation:
        """
        Common evaluation logic for a single miner

        Args:
            miner: Miner instance
            payload_extra: Additional payload parameters

        Returns:
            Evaluation result
        """

        # Build payload
        payload = self.evaluator_config.to_payload(miner)
        if payload_extra:
            payload.update(payload_extra)

        # Execute evaluation with async executor
        try:
            proxy_timeout = (
                self.sandbox_config.proxy_timeout
                if self.env_type == EnvType.AFFINE
                else self.sandbox_config.proxy_timeout + 600
            )

            # Use dedicated thread pool for blocking sandbox calls
            result = await asyncio.get_event_loop().run_in_executor(
                get_executor(),
                lambda: self._sandbox.proxy.evaluator(_timeout=proxy_timeout, **payload)
            )

            return self._parse_evaluation_result(result, miner, payload_extra)

        except asyncio.TimeoutError as e:
            logger.error(f"Evaluation timeout for {self.env_name}: {e}, score set 0")
            return self._create_error_evaluation(e, miner, payload_extra)
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


# ========================= Environment Implementations =========================


class AffineSDKEnv(BaseSDKEnv):
    """Base class for Affine environments (SAT, ABD, DED, HVM, ELR)"""

    @property
    def env_type(self) -> EnvType:
        return EnvType.AFFINE

    async def evaluate(
        self, miner: Union["Miner", Dict[str, Any]],
        task_id: Union[int, List[int], None] = None,
    ) -> Union["Evaluation", Dict[str, "Evaluation"]]:
        """Evaluate using Affine environment endpoint."""

        # Use the IDs from config or default
        payload_extra = {"ids": [0]}

        async def evaluate_single(m):
            return await self._evaluate_single_miner(m, payload_extra)

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
        miner: Union["Miner", Dict[str, Any]],
        task_id: Union[int, List[int], None] = None,
    ) -> Union["Evaluation", Dict[str, "Evaluation"]]:
        """Evaluate using AgentGym environment endpoint."""

        task_ids = self._normalize_task_ids(task_id)
        payload_extra = {
            "ids": task_ids,
            "max_round": self.max_round,
            "task_id": task_ids,  # Keep for backward compatibility in extra
        }

        async def evaluate_single(m):
            return await self._evaluate_single_miner(m, payload_extra)

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

    @property
    def env_name(self) -> str:
        return "affine:sat"


@register_env(EnvType.AFFINE, "affine:abd")
class ABD(AffineSDKEnv):
    """ABD environment for SDK"""

    @property
    def env_name(self) -> str:
        return "affine:abd"


@register_env(EnvType.AFFINE, "affine:ded")
class DED(AffineSDKEnv):
    """DED environment for SDK"""

    @property
    def env_name(self) -> str:
        return "affine:ded"


@register_env(EnvType.AFFINE, "affine:hvm")
class HVM(AffineSDKEnv):
    """HVM environment for SDK"""

    @property
    def env_name(self) -> str:
        return "affine:hvm"


@register_env(EnvType.AFFINE, "affine:elr")
class ELR(AffineSDKEnv):
    """ELR environment for SDK"""

    @property
    def env_name(self) -> str:
        return "affine:elr"


# AgentGym Environments
@register_env(EnvType.AGENTGYM, "agentgym:alfworld")
class ALFWORLD(AgentGymSDKEnv):
    """ALFWORLD environment for SDK"""
    DEFAULT_DATA_LEN = 2500

    @property
    def env_name(self) -> str:
        return "agentgym:alfworld"


@register_env(EnvType.AGENTGYM, "agentgym:webshop")
class WEBSHOP(AgentGymSDKEnv):
    """WEBSHOP environment for SDK"""

    # Override default max_round for WEBSHOP
    DEFAULT_MAX_ROUND = 10

    @property
    def env_name(self) -> str:
        return "agentgym:webshop"


@register_env(EnvType.AGENTGYM, "agentgym:babyai")
class BABYAI(AgentGymSDKEnv):
    """BABYAI environment for SDK"""
    DEFAULT_DATA_LEN = 4000 # 40

    @property
    def env_name(self) -> str:
        return "agentgym:babyai"


@register_env(EnvType.AGENTGYM, "agentgym:sciworld")
class SCIWORLD(AgentGymSDKEnv):
    """SCIWORLD environment for SDK"""
    DEFAULT_DATA_LEN = 4639

    @property
    def env_name(self) -> str:
        return "agentgym:sciworld"


@register_env(EnvType.AGENTGYM, "agentgym:textcraft")
class TEXTCRAFT(AgentGymSDKEnv):
    """TEXTCRAFT environment for SDK"""
    DEFAULT_DATA_LEN = 582

    @property
    def env_name(self) -> str:
        return "agentgym:textcraft"


# ========================= Factory Functions =========================


def create_env_factory(env_class: Type[BaseSDKEnv], **default_kwargs):
    """Create a factory function for environment"""

    async def factory(**kwargs):
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
