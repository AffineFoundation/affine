from __future__ import annotations
import json
import time
import hashlib
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator, root_validator
import bittensor as bt
from affine.setup import get_env_names

__version__ = "0.0.0"


def _truncate(text: Optional[str], max_len: int = 80) -> str:
    """Truncate text to max_len with ellipsis."""
    return "" if not text else textwrap.shorten(text, width=max_len, placeholder="…")


class Miner(BaseModel):
    """Miner information."""
    
    uid: int
    hotkey: str
    model: Optional[str] = None
    revision: Optional[str] = None
    block: Optional[int] = None
    chute: Optional[Dict[str, Any]] = None
    slug: Optional[str] = None
    weights_shas: Optional[list[str]] = None
    
    @validator('weights_shas', pre=True)
    def convert_weights_shas(cls, v):
        """Convert set to sorted list for JSON serialization."""
        if isinstance(v, set):
            return sorted(list(v))
        return v


class Result(BaseModel):
    """Evaluation result for a miner on a specific environment."""
    
    version: str = __version__
    signature: str = ""
    hotkey: str = ""
    
    # Miner info
    miner: Miner
    
    # Evaluation details
    env: str
    score: float
    latency_seconds: float
    success: bool
    error: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    
    @validator("env")
    def validate_env(cls, value):
        """Validate environment name."""
        if value not in get_env_names():
            raise ValueError(f"Unknown environment: '{value}'")
        return value
    
    def _get_sign_data(self) -> str:
        """Get the data string to be signed/verified.
        
        Returns canonical representation of essential fields:
        env:miner_json:score:latency_seconds:timestamp:extra_json
        """
        miner_dict = self.miner.model_dump()
        miner_str = json.dumps(miner_dict, sort_keys=True)
        extra_str = json.dumps(self.extra, sort_keys=True)
        return f"{self.env}:{miner_str}:{self.score:.6f}:{self.latency_seconds:.6f}:{int(self.timestamp)}:{extra_str}"
    
    def sign(self, wallet):
        """Sign the result with wallet.
        
        Signs the evaluation data including challenge details from extra field.
        """
        self.hotkey = wallet.hotkey.ss58_address
        sign_data = self._get_sign_data()
        self.signature = wallet.hotkey.sign(data=sign_data).hex()
    
    def verify(self) -> bool:
        """Verify the result signature."""
        try:
            keypair = bt.Keypair(ss58_address=self.hotkey)
            sign_data = self._get_sign_data()
            signature_bytes = bytes.fromhex(self.signature)
            return keypair.verify(data=sign_data, signature=signature_bytes)
        except Exception:
            return False
    
    @classmethod
    def from_legacy(cls, legacy_data: Dict[str, Any]) -> "Result":
        """Convert legacy Result format to new format.
        
        Legacy format had separate Challenge, Response, Evaluation objects.
        This method extracts the relevant fields and creates a new Result.
        """
        # Extract miner info
        miner_data = legacy_data.get("miner", {})
        miner = Miner(**miner_data) if isinstance(miner_data, dict) else miner_data
        
        # Extract challenge info
        challenge = legacy_data.get("challenge", {})
        env = challenge.get("env", "unknown")
        extra = challenge.get("extra", {})
        
        # Extract response info
        response = legacy_data.get("response", {})
        latency_seconds = response.get("latency_seconds", 0.0)
        success = response.get("success", False)
        error = response.get("error")
        
        # Extract evaluation info
        evaluation = legacy_data.get("evaluation", {})
        score = evaluation.get("score", 0.0)
        eval_extra = evaluation.get("extra", {})
        
        # Merge extra fields
        merged_extra = {**extra, **eval_extra}
        
        # Get timestamp (prefer response timestamp, fallback to challenge timestamp)
        timestamp = response.get("timestamp") or challenge.get("timestamp") or time.time()
        
        return cls(
            version=legacy_data.get("version", __version__),
            signature=legacy_data.get("signature", ""),
            hotkey=legacy_data.get("hotkey", ""),
            miner=miner,
            env=env,
            score=score,
            latency_seconds=latency_seconds,
            success=success,
            error=error,
            extra=merged_extra,
            timestamp=timestamp
        )
    
    class Config:
        arbitrary_types_allowed = True
    
    def json(self, **kwargs):
        return json.dumps(self.dict(**kwargs))
    
    def __repr__(self):
        return (
            f"<Result miner.uid={self.miner.uid} "
            f"env={self.env} "
            f"score={self.score:.4f} "
            f"hotkey={self.hotkey}>"
        )
    
    __str__ = __repr__


class CompactResult(BaseModel):
    """Lightweight result containing only fields needed for weight calculation.
    
    This model significantly reduces memory usage by excluding large text fields
    and error messages that are not used in scoring.
    
    Provides a compatible interface with Result by exposing nested attributes
    through properties (miner.*, etc.).
    """
    
    hotkey: str
    uid: int
    model: Optional[str] = None
    revision: Optional[str] = None
    block: Optional[int] = None
    env: str
    score: float
    
    @classmethod
    def from_result(cls, result: "Result") -> "CompactResult":
        """Create CompactResult from full Result object."""
        return cls(
            hotkey=result.miner.hotkey,
            uid=result.miner.uid,
            model=result.miner.model,
            revision=result.miner.revision,
            block=result.miner.block,
            env=result.env,
            score=result.score
        )
    
    @property
    def miner(self):
        """Provide miner-like interface for compatibility."""
        class _Miner:
            def __init__(self, parent):
                self.hotkey = parent.hotkey
                self.uid = parent.uid
                self.model = parent.model
                self.revision = parent.revision
                self.block = parent.block
        return _Miner(self)
    
    @property
    def challenge(self):
        """Provide challenge-like interface for compatibility."""
        class _Challenge:
            def __init__(self, parent):
                self.env = parent.env
        return _Challenge(self)
    
    @property
    def evaluation(self):
        """Provide evaluation-like interface for compatibility."""
        class _Evaluation:
            def __init__(self, parent):
                self.score = parent.score
        return _Evaluation(self)
    
    class Config:
        arbitrary_types_allowed = True


# Legacy models for backward compatibility (kept minimal, deprecated)
class Challenge(BaseModel):
    """DEPRECATED: Legacy Challenge model. Use Result directly."""
    env: str
    prompt: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)
    challenge_id: Optional[str] = None
    timestamp: Optional[float] = Field(default_factory=time.time)


class Response(BaseModel):
    """DEPRECATED: Legacy Response model. Use Result directly."""
    response: Optional[str] = None
    latency_seconds: float
    attempts: int = 1
    model: str
    error: Optional[str] = None
    success: bool
    timestamp: Optional[float] = Field(default_factory=time.time)


class Evaluation(BaseModel):
    """DEPRECATED: Legacy Evaluation model. Use Result directly."""
    env: str
    score: float
    extra: Dict[str, Any] = Field(default_factory=dict)