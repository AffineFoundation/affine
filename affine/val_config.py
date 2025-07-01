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
ELO_K_FACTOR = 32  # K-factor for Elo rating calculations
CHALLENGES_PER_GAME = 2  # Number of challenges per validation game
RETRY_DELAY = 5.0  # Delay between validation retries in seconds
MAX_MINERS_PER_BATCH = 64  # Maximum number of miners to process in a validation batch
MAX_MATCHES_PER_BATCH = 32  # Maximum number of matches to run in a validation batch

# Sample generation configurations
MAX_GENERATION_ATTEMPTS_MULTIPLIER = 5  # Multiplier for max sample generation attempts
STOCK_REPLENISH_THRESHOLD = 0.8  # Threshold ratio to trigger stock replenishment

# Network configurations
BITTENSOR_NETUID = 120  # Bittensor network UID 