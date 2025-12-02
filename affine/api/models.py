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
    """Extra data containing conversation and request.
    
    Note: conversation is a list of message dicts, not a single dict.
    Each message has structure: {"role": "user/assistant", "content": "..."}
    """

    conversation: Optional[List[Dict[str, Any]]] = None
    request: Optional[Dict[str, Any]] = None


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

    task: Optional[Dict[str, Any]] = None


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
