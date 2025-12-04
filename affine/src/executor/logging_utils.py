"""
Executor Logging Utilities - File-lock-based logging for multiprocess safety
"""

import fcntl
from affine.core.setup import logger


def safe_log(message: str, level: str = "INFO"):
    """Thread-safe logging using file lock on stdout.
    
    Args:
        message: Log message
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    try:
        fcntl.flock(1, fcntl.LOCK_EX)
        try:
            if level == "DEBUG":
                logger.debug(message)
            elif level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            elif level == "ERROR":
                logger.error(message)
            else:
                logger.info(message)
        finally:
            fcntl.flock(1, fcntl.LOCK_UN)
    except Exception:
        pass