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
class TaskCreateRequest(BaseModel):
    """Request to create a task."""

    miner_hotkey: str
    model_revision: str
    model: str
    env: str
    validator_hotkey: str = ""  # Optional, defaults to empty (scheduler will set it)
    priority: int = 0  # Task priority (higher = more urgent)


class TaskCreateResponse(BaseModel):
    """Response after creating a task."""

    task_id: str
    status: str
    created_at: int


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


# TaskCompleteRequest moved above (line 207)
# TaskFailRequest removed (deprecated)
# TaskQueueStatsResponse removed (deprecated, use TaskPoolStatsResponse)


class TaskFetchResponse(BaseModel):
    """Response from task fetch endpoint."""

    task: Optional[Dict[str, Any]] = None


class TaskCompleteRequest(BaseModel):
    """Request to complete a task."""

    task_uuid: str
    success: bool
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class TaskCompleteResponse(BaseModel):
    """Response from task completion."""

    task_uuid: str
    status: str  # 'completed', 'failed', 'not_found', 'error'
    message: str
    timestamp: int


class TaskPoolStatsResponse(BaseModel):
    """Task pool statistics."""

    environments: Dict[str, Dict[str, int]]
    total_pending: int
    total_locked: int
    total_assigned: int
    total_failed: int
    lock_details: Optional[List[Dict[str, Any]]] = None


# Miner Metadata
class SamplingSpeed(BaseModel):
    """Sampling speed statistics."""

    last_hour: float
    last_day: float
    last_week: float


class EnvironmentStats(BaseModel):
    """Statistics by environment."""

    success: int
    failure: int


class RecentError(BaseModel):
    """Recent error details."""

    timestamp: int
    error_type: str
    error_message: str


class MinerStatistics(BaseModel):
    """Miner statistics."""

    total_samples: int
    success_count: int
    error_count: int
    consecutive_errors: int
    sampling_speed: SamplingSpeed
    by_environment: Dict[str, EnvironmentStats]
    recent_errors: List[RecentError]


class MinerStatusResponse(BaseModel):
    """Miner status and metadata."""

    miner_hotkey: str
    model_revision: str
    model: str
    chutes_status: str
    is_paused: bool
    pause_until: Optional[int] = None
    statistics: MinerStatistics
    last_updated: int


class MinerPauseRequest(BaseModel):
    """Request to pause a miner."""

    duration_seconds: Optional[int] = None
    reason: str


class MinerPauseResponse(BaseModel):
    """Response after pausing a miner."""

    miner_hotkey: str
    is_paused: bool
    pause_until: Optional[int] = None
    reason: str


class MinerUnpauseResponse(BaseModel):
    """Response after unpausing a miner."""

    miner_hotkey: str
    is_paused: bool
    pause_until: Optional[int] = None


# MinerMetadataUpdateRequest and MinerMetadataUpdateResponse removed.
# Miner metadata is now queried directly from bittensor metagraph,
# not stored in database.


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


class ConsecutiveErrorsResponse(BaseModel):
    """Consecutive errors check response."""

    miner_hotkey: str
    consecutive_errors: int
    threshold: int
    should_pause: bool
    recent_errors: List[RecentError]


class MinerIsPausedResponse(BaseModel):
    """Response for is-paused check."""

    miner_hotkey: str
    is_paused: bool
    paused_until: Optional[int] = None
    reason: Optional[str] = None


class MinerStatsResponse(BaseModel):
    """Response for miner statistics (used for completion round detection)."""

    miner_hotkey: str
    model_revision: str
    env: str
    total_samples: int
    unique_task_ids: List[int]
    dataset_length: int
    completion_percentage: float


# Admin Operations
class ProtectMinerRequest(BaseModel):
    """Request to protect a miner from retention cleanup."""

    miner_hotkey: str
    reason: str


class ProtectMinerResponse(BaseModel):
    """Response after protecting a miner."""

    miner_hotkey: str
    protected: bool
    reason: str
    protected_at: int


class UnprotectMinerResponse(BaseModel):
    """Response after removing protection."""

    miner_hotkey: str
    protected: bool
    message: str


class CleanupRequest(BaseModel):
    """Request to trigger cleanup."""

    retention_days: int = 90
    dry_run: bool = False


class CleanupResponse(BaseModel):
    """Response after cleanup operation."""

    deleted_count: int
    protected_count: int
    dry_run: bool
    message: str


# Task Generation Models
class TaskGenerationRequest(BaseModel):
    """Request to trigger task generation."""

    envs: Optional[List[str]] = None  # If None, use all configured environments
    max_tasks_per_miner_env: int = Field(
        default=100,
        description="Maximum tasks to create per miner/env combination"
    )


class MinerInfo(BaseModel):
    """Miner information for task generation."""
    
    hotkey: str
    model_revision: str
    model: str
    uid: int = -1
    chute_id: str


class TaskGenerationResponse(BaseModel):
    """Response after task generation."""

    total_tasks_created: int
    tasks_by_env: Dict[str, int]
    miners_processed: int
    errors: List[str]
    timestamp: int


class TaskCleanupResponse(BaseModel):
    """Response after cleaning up invalid tasks."""

    removed_count: int
    timestamp: int


class MinerCompletionStatus(BaseModel):
    """Completion status for a miner."""

    hotkey: str
    model_revision: str
    environments: Dict[str, Dict[str, Any]]  # env -> status


class ActiveMinersResponse(BaseModel):
    """Response with list of active miners."""

    miners: List[MinerInfo]
    total: int