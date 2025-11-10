import os
from dataclasses import dataclass, field


@dataclass
class SamplingConfig:
    """Sampling scheduler configuration"""
    
    # Core parameters - affect system performance and resource consumption
    
    workers_per_env: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_WORKERS_PER_ENV", "10"))
    )
    """Number of concurrent worker threads per environment"""
    
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
    """Interval (seconds) to refresh miner list and adjust sampling rates"""
    
    # Advanced parameters - fine-tuning for specific scenarios
    
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_BATCH_SIZE", "30"))
    )
    """Task batch size for fair scheduling between miners"""
    
    chutes_error_pause_seconds: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_CHUTES_ERROR_PAUSE_SECONDS", "600"))
    )
    """Pause duration (seconds) after consecutive Chutes service errors"""
    
    sink_max_wait: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_SINK_MAX_WAIT", "300"))
    )
    """Maximum wait time (seconds) before uploading incomplete result batches"""
    
    low_sample_threshold: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_LOW_SAMPLE_THRESHOLD", "200"))
    )
    """Sample count threshold below which miners get accelerated sampling"""
    
    low_sample_multiplier: float = field(
        default_factory=lambda: float(os.getenv("AFFINE_LOW_SAMPLE_MULTIPLIER", "3.0"))
    )
    """Sampling rate multiplier for low-sample miners"""
    
    max_consecutive_errors: int = field(
        default_factory=lambda: int(os.getenv("AFFINE_MAX_CONSECUTIVE_ERRORS", "3"))
    )
    """Maximum consecutive Chutes errors before pausing miner"""
    
    # Auto-derived parameters - calculated from core configuration
    
    @property
    def queue_warning_threshold(self) -> int:
        """Queue size threshold for warning logs (auto: 50% of max_size_per_env)"""
        return int(self.queue_max_size_per_env * 0.5)
    
    @property
    def queue_pause_threshold(self) -> int:
        """Queue size threshold for pausing task production (auto: 75% of max_size_per_env)"""
        return int(self.queue_max_size_per_env * 0.75)
    
    @property
    def queue_resume_threshold(self) -> int:
        """Queue size threshold for resuming task production (auto: 25% of max_size_per_env)"""
        return int(self.queue_max_size_per_env * 0.25)
    
    @property
    def batch_flush_interval(self) -> int:
        """Interval (seconds) to flush incomplete batches (auto: 1.5x batch_size)"""
        return int(self.batch_size * 1.5)