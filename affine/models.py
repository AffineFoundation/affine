from __future__ import annotations
import json
import time
import hashlib
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator, root_validator
import bittensor as bt
from affine.setup import ENVS

__version__ = "0.0.0"


def _truncate(text: Optional[str], max_len: int = 80) -> str:
    """Truncate text to max_len with ellipsis."""
    return "" if not text else textwrap.shorten(text, width=max_len, placeholder="…")


class Challenge(BaseModel):
    """Challenge specification for evaluation."""
    
    env_name: str
    prompt: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    challenge_id: Optional[str] = None
    timestamp: Optional[float] = Field(default_factory=time.time)
    
    @validator("env_name")
    def validate_env(cls, value):
        """Validate environment name."""
        if value not in ENVS:
            raise ValueError(f"Unknown environment: '{value}'")
        return value
    
    def json(self, **kwargs):
        return json.dumps(self.dict(**kwargs))
    
    def __repr__(self):
        return f"<Challenge env={self.env_name!r} prompt={_truncate(self.prompt)!r}>"
    
    __str__ = __repr__


class Evaluation(BaseModel):
    """Evaluation result from running a challenge."""
    
    env_name: str
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("env_name")
    def validate_env(cls, value):
        """Validate environment name."""
        if value not in ENVS:
            raise ValueError(f"Unknown environment: '{value}'")
        return value
    
    def json(self, **kwargs):
        return json.dumps(self.dict(**kwargs))
    
    def __repr__(self):
        truncated_extra = {k: _truncate(str(v)) for k, v in self.extra.items()}
        return f"<Evaluation env={self.env_name!r} score={self.score:.4f} extra={truncated_extra!r}>"
    
    __str__ = __repr__


class Response(BaseModel):
    """Response from miner query."""
    
    response: Optional[str]
    latency_seconds: float
    attempts: int
    model: str
    error: Optional[str]
    success: bool
    timestamp: Optional[float] = Field(default_factory=time.time)
    
    def __repr__(self):
        return (
            f"<Response model={self.model!r} success={self.success} "
            f"latency={self.latency_seconds:.3f}s attempts={self.attempts} "
            f"response={_truncate(self.response)!r} error={_truncate(self.error)!r}>"
        )
    
    __str__ = __repr__


class Miner(BaseModel):
    """Miner information."""
    
    uid: int
    hotkey: str
    model: Optional[str] = None
    revision: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None
    slug: Optional[str] = None
    weights_shas: Optional[set[str]] = None


class Result(BaseModel):
    """Complete evaluation result including miner, challenge, response, and evaluation."""
    
    version: str = __version__
    signature: str = ""
    hotkey: str = ""
    miner: Miner
    challenge: Challenge
    response: Response
    evaluation: Evaluation
    
    def sign(self, wallet):
        """Sign the result with wallet."""
        self.hotkey = wallet.hotkey.ss58_address
        challenge_str = str(self.challenge)
        self.signature = wallet.hotkey.sign(data=challenge_str).hex()
    
    def verify(self) -> bool:
        """Verify the result signature."""
        keypair = bt.Keypair(ss58_address=self.hotkey)
        challenge_str = str(self.challenge)
        signature_bytes = bytes.fromhex(self.signature)
        return keypair.verify(data=challenge_str, signature=signature_bytes)
    
    class Config:
        arbitrary_types_allowed = True
    
    def json(self, **kwargs):
        return json.dumps(self.dict(**kwargs))
    
    def __repr__(self):
        return (
            f"<Result miner.uid={self.miner.uid} "
            f"env={self.challenge.env_name} "
            f"score={self.evaluation.score:.4f}>"
        )
    
    __str__ = __repr__