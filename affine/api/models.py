"""
API Request/Response Models

Pydantic models for request validation and response serialization.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# Common models
class ErrorDetail(BaseModel):
    """Error detail in responses."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: int
    request_id: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail


class PaginationInfo(BaseModel):
    """Pagination metadata."""

    total: Optional[int] = None
    limit: int
    next_cursor: Optional[str] = None


# Health & Status
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: int
    version: str
    database: str
    uptime_seconds: int


# Sample Results
class ExtraData(BaseModel):
    """Extra data containing conversation and request.
    
    Note: conversation is a list of message dicts, not a single dict.
    Each message has structure: {"role": "user/assistant", "content": "..."}
    """

    conversation: Optional[List[Dict[str, Any]]] = None
    request: Optional[Dict[str, Any]] = None


class SampleSubmitRequest(BaseModel):
    """Request to submit a sample result.
    
    Note: uid field removed from storage schema.
    Validators should query uid via bittensor metadata using hotkey.
    """

    miner_hotkey: str
    model_revision: str
    model: str
    env: str
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    latency_ms: int
    extra: ExtraData
    validator_hotkey: str
    block_number: int
    signature: str

    @field_validator("model_revision")
    @classmethod
    def validate_revision_format(cls, v: str) -> str:
        """Validate model revision is not empty."""
        if not v or v.strip() == "":
            raise ValueError("model_revision cannot be empty")
        return v


class SampleSubmitResponse(BaseModel):
    """Response after submitting a sample."""

    task_id: str
    created_at: int
    message: str


class SampleDetail(BaseModel):
    """Sample result details."""

    timestamp: int
    env: str
    task_id: str
    score: float
    latency_ms: int
    signature: str
    extra: Optional[ExtraData] = None


class SampleListResponse(BaseModel):
    """List of samples with pagination."""

    samples: List[SampleDetail]
    pagination: PaginationInfo


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


# Task Queue
# TaskCreateRequest and TaskCreateResponse removed - tasks created via batch endpoint only


class TaskDetail(BaseModel):
    """Task details."""

    task_id: str  # Dataset index as string for API compatibility
    miner_hotkey: str
    model_revision: str
    model: str
    env: str
    status: str
    created_at: int
    task_uuid: Optional[str] = None  # UUID from queue for task completion


class TaskListResponse(BaseModel):
    """List of tasks."""

    tasks: List[TaskDetail]


class TaskStatusResponse(BaseModel):
    """Task status update response."""

    task_id: str
    status: str
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None


class TaskFetchResponse(BaseModel):
    """Response from task fetch endpoint."""

    task: Optional[Dict[str, Any]] = None


# Scores
class ScoresByLayer(BaseModel):
    """Scores by layer."""

    layer_3: Optional[float] = Field(None, alias="3")
    layer_4: Optional[float] = Field(None, alias="4")
    layer_5: Optional[float] = Field(None, alias="5")
    layer_6: Optional[float] = Field(None, alias="6")
    layer_7: Optional[float] = Field(None, alias="7")
    layer_8: Optional[float] = Field(None, alias="8")

    class Config:
        populate_by_name = True


class MinerScore(BaseModel):
    """Score details for a miner."""

    miner_hotkey: str
    uid: int
    model_revision: str
    overall_score: float
    average_score: float
    confidence_interval: List[float]
    scores_by_layer: Dict[str, float]
    scores_by_env: Dict[str, float]
    total_samples: int
    is_eligible: bool
    meets_criteria: bool


class ScoresResponse(BaseModel):
    """Scores snapshot response."""

    block_number: int
    calculated_at: int
    scores: List[MinerScore]


class MinerScoreHistory(BaseModel):
    """Historical score for a miner."""

    block_number: int
    calculated_at: int
    overall_score: float
    average_score: float
    confidence_interval: List[float]


class MinerScoreHistoryResponse(BaseModel):
    """Miner score history response."""

    miner_hotkey: str
    history: List[MinerScoreHistory]


# System Configuration
class ConfigParameter(BaseModel):
    """Configuration parameter."""

    name: str
    value: Any
    description: str
    version: int
    updated_at: int


class ConfigListResponse(BaseModel):
    """List of configuration parameters."""

    parameters: List[ConfigParameter]


class ConfigUpdateRequest(BaseModel):
    """Request to update a configuration parameter."""

    value: Any
    description: str


class ConfigUpdateResponse(BaseModel):
    """Response after updating a parameter."""

    name: str
    value: Any
    version: int
    updated_at: int
    message: str


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
