"""Error classification utilities for sampling scheduler.

Centralizes error pattern matching to distinguish between:
1. Chutes service errors (should pause sampling and skip results)
2. Model capability errors (should record as valid results with score=0)
"""

from typing import Optional


# Chutes service errors that should pause sampling and skip results
# These indicate infrastructure/service issues, not model performance
CHUTES_SERVICE_ERROR_PATTERNS = [
    "Invalid API key",           # Miner API key misconfigured (401)
    "No instances available",    # Chutes instance not started or unavailable
    "HTTP 503",                  # Chutes service unavailable
    "HTTP 500",                  # Chutes internal server error
    "HTTP 429",                  # Rate limit (too many requests)
    "HTTP 402",                  # Chute creator has insufficient balance
    "Error code: 429",           # OpenAI rate limit error
    "Error code: 402",           # OpenAI insufficient balance
    "Error code: 401",           # OpenAI auth failed (invalid API key)
    "Error code: 503",           # OpenAI service unavailable
    "Error code: 500",           # OpenAI internal error
    "CHUTES_API_KEY",            # Chutes API key env var missing
    "maximum capacity",          # Chutes reached max capacity
    "try again later",           # Service busy, retry suggested
    "zero balance",              # Chute creator has zero balance
]

# Model capability errors that should be recorded as valid results
# These indicate the model failed the task, which is valid evaluation data
MODEL_ERROR_PATTERNS = [
    "HTTP 400",                  # Bad request (model capability issue)
    "400 Bad Request",           # Bad request (model capability issue)
]


def is_service_error(error_msg: Optional[str]) -> bool:
    """Detect Chutes service errors that should pause sampling.
    
    Args:
        error_msg: Error message to classify
        
    Returns:
        True if error is a service error, False otherwise
        
    Note:
        HTTP 400 is intentionally excluded - it indicates model capability 
        issues which should be recorded as valid results (score=0) rather 
        than service errors.
    """
    if not error_msg:
        return False
    return any(pattern in error_msg for pattern in CHUTES_SERVICE_ERROR_PATTERNS)


def is_model_error(error_msg: Optional[str]) -> bool:
    """Detect model capability errors that should be recorded as valid results.
    
    Args:
        error_msg: Error message to classify
        
    Returns:
        True if error is a model capability error, False otherwise
        
    Note:
        Model errors indicate the model failed to handle the request properly,
        which is valid evaluation data showing model limitations. These should
        be recorded with score=0 rather than being skipped.
    """
    if not error_msg:
        return False
    return any(pattern in error_msg for pattern in MODEL_ERROR_PATTERNS)