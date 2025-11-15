"""
DAO implementations for all tables

Provides high-level data access interfaces.
"""

from affine.database.dao.sample_results import SampleResultsDAO
from affine.database.dao.task_queue import TaskQueueDAO
from affine.database.dao.execution_logs import ExecutionLogsDAO
from affine.database.dao.scores import ScoresDAO
from affine.database.dao.system_config import SystemConfigDAO
from affine.database.dao.data_retention import DataRetentionDAO

__all__ = [
    "SampleResultsDAO",
    "TaskQueueDAO",
    "ExecutionLogsDAO",
    "ScoresDAO",
    "SystemConfigDAO",
    "DataRetentionDAO",
]