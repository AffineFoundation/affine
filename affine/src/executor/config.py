"""
Executor configuration for different environments
"""

# Max concurrent tasks for each environment
ENV_MAX_CONCURRENT = {
    "affine:sat": 30,
    "affine:abd": 30,
    "affine:ded": 30,
    "affine:ded-v2": 180,
    "affine:abd-v2": 120,
    "agentgym:alfworld": 60,
    "agentgym:webshop": 60,
    "agentgym:babyai": 60,
    "agentgym:sciworld": 60,
    "agentgym:textcraft": 60,
}

# Default max concurrent tasks if environment not found in config
DEFAULT_MAX_CONCURRENT = 30


def get_max_concurrent(env: str) -> int:
    """Get max concurrent tasks for a specific environment.
    
    Args:
        env: Environment name (e.g., "affine:sat", "agentgym:webshop")
        
    Returns:
        Max concurrent tasks for the environment
    """
    return ENV_MAX_CONCURRENT.get(env, DEFAULT_MAX_CONCURRENT)