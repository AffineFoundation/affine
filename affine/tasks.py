#!/usr/bin/env python3

import os
import traceback
import random
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, Field, validator, ValidationError

from .quixand.core.sandbox_manager import get_sandbox
from . import Miner

logger = logging.getLogger(__name__)


class Evaluation(BaseModel):
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)


class BaseSDKEnv(ABC):
    """Base class for all SDK environments"""

    def __init__(self):
        super().__init__()
        self._sandbox = None
        self._sandbox_lock = None

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Return environment name"""
        pass

    @property
    @abstractmethod
    def template_name(self) -> str:
        """Return template name for sandbox creation"""
        pass


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


class AffineSDKEnv(BaseSDKEnv):
    """Base class for Affine environments (SAT, ABD, DED, HVM, ELR)"""

    @property
    def template_name(self) -> str:
        return f"affine:{self.env_name}"

    async def evaluate(self, miner: Union["Miner", Dict[str, Any]]) -> Union["Evaluation", Dict[str, "Evaluation"]]:
        """Evaluate using Affine environment endpoint.
        
        Args:
            miner: A single Miner object or a dict[str, Miner].
        Returns:
            If input is a single Miner -> Evaluation
            If input is a dict of miners -> dict[str, Evaluation]
        """
        import traceback

        sbx_env = {
            "NO_PROXY": "localhost,127.0.0.1",
            "PYTHONPATH": "/app",
        }

        async def _evaluate_single(m):
            """Internal helper to evaluate one miner"""
            sandbox = get_sandbox(
                template=self.template_name,
                shared=True,
                timeout=3600,
                env=sbx_env,
            )

            payload = {
                "model": m.model,
                "base_url": f"https://{m.slug}.chutes.ai/v1",
                "temperature": 0.7,
                "timeout": 600,
                "ids": [0],
            }

            try:
                result = await asyncio.to_thread(
                    lambda: sandbox.proxy.evaluator(_timeout=700, **payload)
                )

                total_score = float(result.get("total_score", 0.0))
                success_rate = float(result.get("success_rate", 0.0))
                details = result.get("details", [{}])[0]

                return Evaluation(
                    score=total_score,
                    extra={
                        "success": bool(success_rate > 0),
                        "details": details,
                        "miner": m,
                    },
                )
            except BaseException as e:
                logger.error(f"Evaluation failed for {self.env_name}: {e}")
                traceback.print_exc()
                return Evaluation(
                    score=0.0,
                    extra={"success": False, "error": str(e), "miner": m},
                )

        if isinstance(miner, dict):
            results = {}
            for key, m in miner.items():
                if not hasattr(m, "model") or not hasattr(m, "slug"):
                    logger.warning(f"Skipping invalid miner entry: {key}")
                    continue
                results[key] = await _evaluate_single(m)
            return results
        else:
            return await _evaluate_single(miner)


class AgentGymSDKEnv(BaseSDKEnv):
    """Base class for AgentGym environments"""

    def __init__(self, data_len: int = 200, max_round: int = 10):
        super().__init__()
        self.data_len = data_len
        self.max_round = max_round

    @property
    def template_name(self) -> str:
        return f"agentgym:{self.env_name}"
    async def evaluate(
        self, miner: Union["Miner", Dict[str, Any]], task_id: Union[int, List[int], None] = None
    ) -> Union["Evaluation", Dict[str, "Evaluation"]]:
        """Evaluate using AgentGym environment endpoint.

        Args:
            miner: A single Miner instance or a dict of miners.
            task_id: Optional task index(es).
        Returns:
            If input is a single Miner → Evaluation
            If input is a dict of miners → dict[str, Evaluation]
        """
        sbx_env = {
            "NO_PROXY": "localhost,127.0.0.1",
            "PYTHONPATH": "/app",
        }

        async def _evaluate_single(miner_obj, task_id_list):
            sandbox = get_sandbox(
                template=self.template_name,
                shared=True,
                timeout=3600,
                env=sbx_env,
            )

            payload = {
                "model": miner_obj.model,
                "base_url": f"https://{miner_obj.slug}.chutes.ai/v1",
                "temperature": 0.7,
                "ids": task_id_list,
                "max_round": self.max_round,
                "timeout": 1200,
            }

            try:
                result = await asyncio.to_thread(
                    lambda: sandbox.proxy.evaluator(_timeout=1300, **payload)
                )

                total_score = float(result.get("total_score", 0.0))
                success_rate = float(result.get("success_rate", 0.0))
                details = result.get("details", [{}])[0]

                return Evaluation(
                    score=total_score,
                    extra={
                        "success": bool(success_rate > 0),
                        "details": details,
                        "task_id": task_id_list,
                        "miner": miner_obj,
                    },
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {self.env_name}: {e}")
                return Evaluation(
                    score=0.0,
                    extra={
                        "success": False,
                        "error": str(e),
                        "task_id": task_id_list,
                        "miner": miner_obj,
                    },
                )

        # normalize task_id
        if task_id is None:
            task_id = [random.randint(0, int(self.data_len) - 1)]
        elif isinstance(task_id, int):
            task_id = [task_id]
        elif isinstance(task_id, list):
            if len(task_id) == 0:
                task_id = [random.randint(0, int(self.data_len) - 1)]
        else:
            raise TypeError(f"task_id must be int, list[int], or None, got {type(task_id)}")

        # branch: single Miner or dict of miners
        if isinstance(miner, dict):
            results = {}
            for key, m in miner.items():
                if not hasattr(m, "model") or not hasattr(m, "slug"):
                    logger.warning(f"Skipping invalid miner entry: {key}")
                    continue
                results[key] = await _evaluate_single(m, task_id)
            return results
        else:
            return await _evaluate_single(miner, task_id)


# Concrete environment implementations


class SAT(AffineSDKEnv):
    """SAT environment for SDK"""

    @property
    def env_name(self) -> str:
        return "sat"


class ABD(AffineSDKEnv):
    """ABD environment for SDK"""

    @property
    def env_name(self) -> str:
        return "abd"


class DED(AffineSDKEnv):
    """DED environment for SDK"""

    @property
    def env_name(self) -> str:
        return "ded"


class HVM(AffineSDKEnv):
    """HVM environment for SDK"""

    @property
    def env_name(self) -> str:
        return "hvm"


class ELR(AffineSDKEnv):
    """ELR environment for SDK"""

    @property
    def env_name(self) -> str:
        return "elr"


class ALFWORLD(AgentGymSDKEnv):
    """ALFWORLD environment for SDK"""

    @property
    def env_name(self) -> str:
        return "alfworld"


class WEBSHOP(AgentGymSDKEnv):
    """WEBSHOP environment for SDK"""

    @property
    def env_name(self) -> str:
        return "webshop"


class BABYAI(AgentGymSDKEnv):
    """BABYAI environment for SDK"""

    @property
    def env_name(self) -> str:
        return "babyai"


class SCIWORLD(AgentGymSDKEnv):
    """SCIWORLD environment for SDK"""

    @property
    def env_name(self) -> str:
        return "sciworld"


class TEXTCRAFT(AgentGymSDKEnv):
    """TEXTCRAFT environment for SDK"""

    @property
    def env_name(self) -> str:
        return "textcraft"


# Convenience functions matching the SDK API
async def SAT_factory() -> SAT:
    """Create SAT environment"""
    return SAT()


async def ABD_factory() -> ABD:
    """Create ABD environment"""
    return ABD()


async def DED_factory() -> DED:
    """Create DED environment"""
    return DED()


async def HVM_factory() -> HVM:
    """Create HVM environment"""
    return HVM()


async def ELR_factory() -> ELR:
    """Create ELR environment"""
    return ELR()


async def ALFWORLD_factory(**kwargs) -> ALFWORLD:
    """Create ALFWORLD environment"""
    return ALFWORLD(**kwargs)


async def WEBSHOP_factory(**kwargs) -> WEBSHOP:
    """Create WEBSHOP environment"""
    return WEBSHOP(**kwargs)


async def BABYAI_factory(**kwargs) -> BABYAI:
    """Create BABYAI environment"""
    return BABYAI(**kwargs)


async def SCIWORLD_factory(**kwargs) -> SCIWORLD:
    """Create SCIWORLD environment"""
    return SCIWORLD(**kwargs)
