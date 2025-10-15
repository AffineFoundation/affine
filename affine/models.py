from __future__ import annotations
import os
import json
import time
import uuid
import hashlib
import aiohttp
import asyncio
import logging
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator, root_validator, ValidationError
import bittensor as bt
from affine.quixand.core.sandbox_manager import get_sandbox
from affine.setup import ENVS, logger

__version__ = "0.0.0"

def _truncate(t: Optional[str], max_len: int = 80) -> str:
    return "" if not t else textwrap.shorten(t, width=max_len, placeholder="…")

class BaseEnv(BaseModel, ABC):
    class Config: arbitrary_types_allowed = True
    @property
    def name(self) -> str: return self.__class__.__name__
    def __hash__(self):     return hash(self.name)
    def __repr__(self):     return self.name

_SBX_POOL: Dict[str, Any] = {}
_SBX_LOCKS: Dict[str, asyncio.Lock] = {}
_SBX_SEMS: Dict[str, asyncio.Semaphore] = {}

class ContainerEnv(BaseEnv):
    env_name: str
    data_len: int = 200
    max_round: int = 10
    evaluator_timeout: int = 1200
    _round_counter: int = 0

    def _pool_key(self) -> str:
        return self.name

    def _get_lock(self) -> asyncio.Lock:
        lk = _SBX_LOCKS.get(self._pool_key())
        if lk is None:
            lk = asyncio.Lock()
            _SBX_LOCKS[self._pool_key()] = lk
        return lk

    async def _get_shared(self):
        async with self._get_lock():
            sbx = _SBX_POOL.get(self._pool_key())
            if sbx is None:
                logger.info(f"[ENV] Creating sandbox via SandboxManager for {self.name} ENV_NAME={self.env_name}")
                sbx_env = {
                    "NO_PROXY": "localhost,127.0.0.1",
                    "PYTHONPATH": "/app",
                }
                try:
                    sbx = await asyncio.to_thread(
                        get_sandbox,
                        template=self.name,
                        shared=True,
                        timeout=max(int(self.evaluator_timeout) + 900, 1800),
                        env=sbx_env,
                    )
                except Exception as e:
                    logger.error(f"[ENV] Sandbox creation failed for {self.name}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                    raise
                _SBX_POOL[self._pool_key()] = sbx
                try:
                    logger.debug(f"[ENV] Sandbox started for {self.name} id={getattr(sbx, 'id', '?')} container_id={getattr(sbx, 'container_id', '?')}")
                except Exception:
                    pass
                try:
                    await self._health_check(sbx)
                except Exception as e:
                    logger.warning(f"ensure_ready pending for {self.name}: {e}")
            sem = _SBX_SEMS.get(self._pool_key())
            if sem is None:
                sem = asyncio.Semaphore(int(os.getenv("AFFINE_ENV_CONCURRENCY", "16")))
                _SBX_SEMS[self._pool_key()] = sem
            return sbx, sem

    async def _health_check(self, sbx):
        try:
            logger.debug(f"[ENV] Health check for {self.name} on port 8000")
            await asyncio.to_thread(lambda: sbx.proxy._health(port=8000, timeout=60))
        except Exception as e:
            raise RuntimeError(f"Sandbox for {self.name} failed health: {e}")

    async def ensure_ready(self):
        reuse = os.getenv("AFFINE_REUSE_CONTAINERS", "1")
        if reuse != "0":
            await self._get_shared()

    async def run_episode(self, policy: "Miner", task_id: Optional[int]) -> "Evaluation":
        sbx, sem = await self._get_shared()
        base_url = f"https://{policy.slug}.chutes.ai/v1" if policy.slug else None
        if not base_url:
            raise RuntimeError("Miner slug/base_url missing")

        env_task_id = task_id
        if env_task_id is None:
            raise ValueError(f"task_id is required for {self.name}; pass a deterministic dataset index")

        payload = {
            "model": policy.model,
            "base_url": base_url,
            "temperature": 0.7,
            "ids": [int(env_task_id)],
            "max_round": int(self.max_round),
            "timeout": int(self.evaluator_timeout),
        }
        start = time.monotonic()
        async with sem:
            try:
                data = await asyncio.to_thread(lambda: sbx.proxy.evaluator(_timeout=self.evaluator_timeout, **payload))
            except Exception as e:
                logger.error(f"[ENV] /evaluator call failed for {self.name}: {type(e).__name__}: {e}")
                return Evaluation(env=self, score=0.0, extra={"error": str(e), "evaluation_failed": True})
        latency = time.monotonic() - start
        total_score = float(data.get("total_score", data.get("score", 0.0)))
        success_rate = float(data.get("success_rate", 0.0))
        num_eval = int(data.get("num_evaluated", data.get("n", 0)))
        extra_payload = {
            "success_rate": success_rate,
            "num_evaluated": num_eval,
            "task_id": int(env_task_id),
            "environment_task_id": int(env_task_id),
        }
        if isinstance(data, dict):
            if "details" in data:   extra_payload["details"] = data.get("details")
            if "task_name" in data: extra_payload["task_name"] = data.get("task_name")
            if "time_taken" in data: extra_payload["time_taken"] = data.get("time_taken")

        logger.info(
            f"[REWARD] U{policy.uid:>3d} {self.name:<20} id={env_task_id} total_score={total_score:.4f} success_rate={success_rate:.3f} n={num_eval} latency={latency:.3f}s"
        )
        return Evaluation(
            env=self,
            score=total_score,
            extra=extra_payload,
        )

class AgentGymContainerEnv(ContainerEnv):
    @property
    def name(self) -> str:
        return f"agentgym:{self.env_name}"

class AffineContainerEnv(ContainerEnv):
    evaluator_timeout: int = 280
    @property
    def name(self) -> str:
        return f"affine:{self.env_name}"

class Challenge(BaseModel):
    env:  BaseEnv
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    challenge_id: Optional[str] = None
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    environment_task_id: Optional[int] = None
    timestamp: Optional[float] = Field(default_factory=time.time)
    @root_validator(pre=True)
    def set_challenge_id(cls, values):
        if "challenge_id" not in values or values["challenge_id"] is None:
            env = values["env"]
            prompt = values["prompt"]
            extra = values.get("extra", {})
            if not isinstance(env, str): env = env.name
            base_dict = { "env": env,"prompt": prompt, "extra": extra}
            canonical = json.dumps(base_dict, sort_keys=True, separators=(",", ":"))
            cid = hashlib.sha256(canonical.encode()).hexdigest()
            values["challenge_id"] = cid
        return values
    @root_validator
    def propagate_task_id(cls, values):
        extra = values.get("extra") or {}
        task_id = values.get("task_id")
        if task_id is not None:
            extra = dict(extra)
            extra["task_id"] = task_id
            values["extra"] = extra
        return values
    @validator("env", pre=True)
    def _parse_env(cls, v):
        if isinstance(v, str):
            name = v.strip()
            if name not in ENVS:
                raise ValueError(f"Inactive env '{name}'")
            if ":" in name:
                prefix, env_name = name.split(":", 1)
                if prefix == "agentgym":
                    return AgentGymContainerEnv(env_name=env_name)
                if prefix == "affine":
                    return AffineContainerEnv(env_name=env_name)
                raise ValueError(f"Unknown env prefix '{prefix}'")
            return AgentGymContainerEnv(env_name=name)
        return v
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    def ensure_environment_task_id(self, data_len: Optional[int] = None) -> Optional[int]:
        """Assign (once) and return the environment-level task identifier.

        Derives a deterministic index from this challenge's UUID so every
        validator maps the prompt to the same dataset slot.
        """
        if data_len is None:
            env = getattr(self, "env", None)
            if env is not None:
                data_len = getattr(env, "data_len", None)
        upper = int(data_len) if data_len else 0
        if upper > 0 and self.environment_task_id is None and self.task_id:
            # Deterministic mapping: UUID -> dataset slot
            self.environment_task_id = int(self.task_id, 16) % upper
        env_id = self.environment_task_id
        meta = dict(self.extra)
        if env_id is not None:
            meta["dataset_id"] = env_id
            meta["environment_task_id"] = env_id
        else:
            meta.pop("dataset_id", None)
            meta.pop("environment_task_id", None)
        self.extra = meta
        return env_id

    def attach_metadata(self, evaluation: "Evaluation", environment_task_id: Optional[int] = None) -> None:
        """Propagate challenge identifiers onto an Evaluation payload."""
        env_id = environment_task_id if environment_task_id is not None else self.environment_task_id
        meta = dict(evaluation.extra)
        if env_id is not None:
            meta["environment_task_id"] = env_id
        if self.task_id:
            meta["task_id"] = self.task_id
        evaluation.extra = meta
    async def evaluate(self, resp: "Response") -> "Evaluation":
        self.ensure_environment_task_id()
        evaluation = await self.env.evaluate(self, resp)
        self.attach_metadata(evaluation)
        return evaluation
    def __repr__(self):
        return f"<Challenge env={self.env.name!r} prompt={_truncate(self.prompt)!r}>"
    __str__ = __repr__

class Evaluation(BaseModel):
    env: BaseEnv
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)
    @validator("env", pre=True)
    def _parse_env(cls, v):
        if isinstance(v, str):
            name = v.strip()
            if name not in ENVS:
                raise ValueError(f"Inactive env '{name}'")
            if ":" in name:
                prefix, env_name = name.split(":", 1)
                if prefix == "agentgym":
                    return AgentGymContainerEnv(env_name=env_name)
                if prefix == "affine":
                    return AffineContainerEnv(env_name=env_name)
                raise ValueError(f"Unknown env prefix '{prefix}'")
            return AgentGymContainerEnv(env_name=name)
        return v
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    def __repr__(self):
        ex = {k: _truncate(str(v)) for k, v in self.extra.items()}
        return f"<Evaluation env={self.env.name!r} score={self.score:.4f} extra={ex!r}>"
    __str__ = __repr__

class Response(BaseModel):
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]
    success: bool
    timestamp: Optional[float] = Field(default_factory=time.time)
    def __repr__(self):
        return (f"<Response model={self.model!r} success={self.success} "
                f"latency={self.latency_seconds:.3f}s attempts={self.attempts} "
                f"response={_truncate(self.response)!r} error={_truncate(self.error)!r}>")
    __str__ = __repr__

class Miner(BaseModel):
    uid: int; hotkey: str; model: Optional[str] = None
    revision: Optional[str] = None; block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None
    slug: Optional[str] = None
    weights_shas: Optional[set[str]] = None

class Result(BaseModel):
    version: str = __version__
    signature: str = ""
    hotkey: str = ""
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation
    def sign(self, wallet):
        self.hotkey = wallet.hotkey.ss58_address
        self.signature = (wallet.hotkey.sign( data = str(self.challenge) )).hex()
    def verify( self ) -> bool:
        return bt.Keypair(ss58_address=self.hotkey).verify( data = str(self.challenge), signature = bytes.fromhex( self.signature) )
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {BaseEnv: lambda v: v.name}
    def json(self, **kw): return json.dumps(self.dict(**kw))
    def __repr__(self): return f"<Result {self.miner.uid=} {self.challenge.env.name=} score={self.evaluation.score:.4f}>"
    __str__ = __repr__
