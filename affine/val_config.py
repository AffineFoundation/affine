"""
Configuration constants for the Affine validation system.

This module centralizes all configuration constants used throughout the codebase.
"""

# Sample management configuration
TARGET_STOCK = 50  # Target number of samples to maintain per environment
MAX_CONCURRENT_REQUESTS = 10  # Maximum number of concurrent sample generation requests

# Timeout configurations
LLM_RESPONSE_TIMEOUT = 600  # Default timeout for LLM API responses in seconds
DEFAULT_PROGRAM_EXECUTION_TIMEOUT = 30  # Default timeout for program execution in seconds

# Validation configurations
# GRPO evaluation parameters
EVALUATOR_EMA_ALPHA = 0.1  # Exponential moving average smoothing factor
EVALUATOR_SKEW_PENALTY_WEIGHT = 0.1  # Weight for domain skew penalty
DEFAULT_SAMPLES = 10  # Default number of samples each miner will face
RETRY_DELAY = 5.0  # Delay between validation retries in seconds
MAX_MINERS_PER_BATCH = 64  # Maximum number of miners to process in a validation batch
MAX_MATCHES_PER_BATCH = 32  # Maximum number of matches to run in a validation batch

# Sample generation configurations
MAX_GENERATION_ATTEMPTS_MULTIPLIER = 5  # Multiplier for max sample generation attempts
STOCK_REPLENISH_THRESHOLD = 0.8  # Threshold ratio to trigger stock replenishment

# Network configurations
BITTENSOR_NETUID = 120  # Bittensor network UID 