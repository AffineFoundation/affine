import os
from dataclasses import dataclass, field


@dataclass
class SamplingConfig:
    """Sampling scheduler configuration"""
    
    daily_rate_per_env: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_DAILY_RATE_PER_ENV", "200"))
    )
    low_sample_threshold: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_LOW_SAMPLE_THRESHOLD", "200"))
    )
    low_sample_multiplier: float = field(
        default_factory=lambda: float(os.getenv("AFFINE_LOW_SAMPLE_MULTIPLIER", "3.0"))
    )
    
    queue_max_size: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_QUEUE_MAX_SIZE", "10000"))
    )
    queue_warning_threshold: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_QUEUE_WARNING_THRESHOLD", "5000"))
    )
    queue_pause_threshold: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_QUEUE_PAUSE_THRESHOLD", "8000"))
    )
    queue_resume_threshold: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_QUEUE_RESUME_THRESHOLD", "3000"))
    )
    
    num_evaluation_workers: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_NUM_EVALUATION_WORKERS", "20"))
    )
    
    chutes_error_pause_seconds: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_CHUTES_ERROR_PAUSE_SECONDS", "600"))
    )
    max_consecutive_errors: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_MAX_CONSECUTIVE_ERRORS", "3"))
    )
    
    miner_refresh_interval: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_MINER_REFRESH_INTERVAL", "600"))
    )
    
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_BATCH_SIZE", "50"))
    )
    batch_flush_interval: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_BATCH_FLUSH_INTERVAL", "30"))
    )
    
    sink_batch_size: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_SINK_BATCH_SIZE", "300"))
    )
    sink_max_wait: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_SINK_MAX_WAIT", "300"))
    )