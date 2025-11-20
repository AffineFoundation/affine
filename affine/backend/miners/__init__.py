"""
Miners Monitor Service

Independent background service for monitoring and validating miners.
"""

from affine.backend.miners_monitor.miners_monitor import MinersMonitor, MinerInfo

__all__ = ['MinersMonitor', 'MinerInfo']