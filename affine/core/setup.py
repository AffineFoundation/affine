import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

NETUID = 120
TRACE = 5

# Add custom TRACE level
logging.addLevelName(TRACE, "TRACE")


def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


logging.Logger.trace = _trace
logger = logging.getLogger("affine")


def _get_component_name() -> str:
    """
    Identify component name from command line arguments.
    Supported components: api, scheduler, executor, monitor, scorer, validator
    """
    # Get component name from command line arguments
    if len(sys.argv) > 1:
        component = sys.argv[-1].lower()
        valid_components = ["api", "scheduler", "executor", "monitor", "scorer", "validator"]
        if component in valid_components:
            return component
    
    # Default to affine
    return "affine"


def _setup_file_handler(component: str, level: int) -> logging.Handler:
    """
    Setup log file handler with rotation for the specified component.
    
    Log directory structure: /var/log/affine/{component}/
    Log file: {component}.log
    Rotation policy: every 3 days at midnight, keep 10 backups
    File suffix format: %Y-%m-%d
    """
    log_dir = Path(f"/var/log/affine/{component}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{component}.log"
    
    # Create TimedRotatingFileHandler
    # when='midnight': rotate at midnight
    # interval=3: every 3 days
    # atTime=None: use midnight (00:00:00) as rotation time
    handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=3,
        backupCount=20,
        encoding="utf-8",
        atTime=None  # Rotate at midnight (00:00:00)
    )
    
    # Set suffix format to date (YYYY-MM-DD)
    handler.suffix = "%Y-%m-%d"
    
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    return handler


def _silence_noisy_loggers():
    """Silence noisy third-party library loggers"""
    noisy_loggers = [
        "websockets",
        "bittensor",
        "bittensor-cli",
        "btdecode",
        "asyncio",
        "aiobotocore.regions",
        "aiobotocore.credentials",
        "botocore",
        "httpx",
        "httpcore",
        "docker",
        "urllib3",
    ]
    
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    # affinetes has its own handler, disable propagation to avoid duplicate logs
    affinetes_logger = logging.getLogger("affinetes")
    affinetes_logger.setLevel(logging.WARNING)
    affinetes_logger.propagate = False


def setup_logging(verbosity: int):
    """
    Setup logging system.
    
    Args:
        verbosity: Log level (0=SILENT, 1=INFO, 2=DEBUG, 3=TRACE)
    """
    # Determine log level
    level_map = {
        0: logging.CRITICAL + 1,  # Silent
        1: logging.INFO,
        2: logging.DEBUG,
        3: TRACE,
    }
    level = level_map.get(verbosity, logging.INFO)
    
    # Get component name
    component = _get_component_name()
    
    # Configure root logger (console output)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True
    )
    
    # Add file handler (log rotation)
    try:
        file_handler = _setup_file_handler(component, level)
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Log file: /var/log/affine/{component}/{component}.log")
    except Exception as e:
        logger.warning(f"Failed to create log file: {e}")
    
    # Silence noisy third-party loggers
    _silence_noisy_loggers()
    
    # Set affine logger level
    logging.getLogger("affine").setLevel(level)
    
    # Set uvicorn logger (only for API service)
    if component == "api":
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def info():
    """Shortcut: Set INFO level logging"""
    setup_logging(1)


def debug():
    """Shortcut: Set DEBUG level logging"""
    setup_logging(2)


def trace():
    """Shortcut: Set TRACE level logging"""
    setup_logging(3)