import os
from dataclasses import dataclass, field


@dataclass
class SamplingConfig:
    """Sampling scheduler configuration - simplified for environment-based global sampling."""
    
    # Core parameters
    
    workers_per_env: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_WORKERS_PER_ENV", "10"))
    )
    """Number of concurrent evaluation workers per environment"""
    
    queue_max_size_per_env: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_QUEUE_MAX_SIZE_PER_ENV", "1000"))
    )
    """Maximum task queue capacity per environment"""
    
    sink_batch_size: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_SINK_BATCH_SIZE", "300"))
    )
    """Batch size for uploading results to storage"""
    
    miner_refresh_interval: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_MINER_REFRESH_INTERVAL", "1800"))
    )
    """Interval (seconds) to refresh miner list from metagraph"""
    
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_BATCH_SIZE", "30"))
    )
    """Task batch size for fair scheduling between miners"""
    
    sink_max_wait: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_SINK_MAX_WAIT", "300"))
    )
    """Maximum wait time (seconds) before uploading incomplete result batches"""
    
    # Error handling
    
    chutes_error_pause_seconds: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_CHUTES_ERROR_PAUSE_SECONDS", "600"))
    )
    """Initial pause duration (seconds) after consecutive service errors"""
    
    max_pause_seconds: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_MAX_PAUSE_SECONDS", "86400"))
    )
    """Maximum pause duration (24 hours) for exponential backoff"""
    
    max_consecutive_errors: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_MAX_CONSECUTIVE_ERRORS", "3"))
    )
    """Number of consecutive errors before pausing miner"""
    
    # Auto-derived parameters
    
    @property
    def queue_warning_threshold(self) -> int:
        """Queue warning threshold (50% of max)"""
        return int(self.queue_max_size_per_env * 0.5)
    
    @property
    def queue_pause_threshold(self) -> int:
        """Queue pause threshold (75% of max)"""
        return int(self.queue_max_size_per_env * 0.75)
    
    @property
    def queue_resume_threshold(self) -> int:
        """Queue resume threshold (25% of max)"""
        return int(self.queue_max_size_per_env * 0.25)
    
    @property
    def batch_flush_interval(self) -> int:
        """Batch flush interval (1.5x batch_size seconds)"""
        return int(self.batch_size * 1.5)