"""
API Request/Response Models

Pydantic models for request validation and response serialization.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class SampleSubmitResponse(BaseModel):
    """Response after submitting a sample."""

    task_id: str
    created_at: int
    message: str


class ExtraData(BaseModel):
    """Extra data with dynamic fields.
    
    Common fields:
    - conversation: List of message dicts with {"role": "user/assistant", "content": "..."}
    - request: Request parameters dict
    - image: Docker image used for evaluation
    
    Note: This model accepts any additional fields dynamically.
    """

    class Config:
        extra = "allow"  # Allow additional fields beyond defined ones

    conversation: Optional[List[Dict[str, Any]]] = None
    request: Optional[Dict[str, Any]] = None
    image: Optional[str] = None


class SampleFullResponse(BaseModel):
    """Full sample details."""

    miner_hotkey: str
    model_revision: str
    env: str
    score: float
    signature: str
    extra: ExtraData
    timestamp: int
    block_number: int


class TaskFetchResponse(BaseModel):
    """Response from task fetch endpoint."""

    tasks: List[Dict[str, Any]] = []


class MinerScore(BaseModel):
    """Score details for a miner."""

    miner_hotkey: str
    uid: int
    model_revision: str
    model: str
    first_block: int
    overall_score: float
    average_score: float
    scores_by_layer: Dict[str, float]
    scores_by_env: Dict[str, Dict[str, Any]]  # Changed to support {env: {score, sample_count}}
    total_samples: int
    is_eligible: bool
    cumulative_weight: Optional[float] = None


class ScoresResponse(BaseModel):
    """Scores snapshot response."""

    block_number: int
    calculated_at: int
    scores: List[MinerScore]


# Execution Logs
class ExecutionLog(BaseModel):
    """Execution log entry."""

    log_id: str
    timestamp: int
    task_id: str
    env: str
    status: str  # 'success' or 'failed'
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: int


class ExecutionLogsResponse(BaseModel):
    """List of execution logs."""

    logs: List[ExecutionLog]
