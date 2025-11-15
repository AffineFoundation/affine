"""
Scorer Data Models

数据模型用于分数计算器的内部数据表示
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time


@dataclass
class MinerScore:
    """单个 Miner 在单个环境的分数"""
    hotkey: str
    uid: int
    model_revision: str
    env: str
    average_score: float
    block_number: int
    is_superior: bool = False
    superiority_reason: str = ""


@dataclass
class TaskIdRange:
    """
    Task ID 范围定义
    
    用于支持数据集扩展：
    - 采样可以使用新范围
    - 权重计算可以指定旧范围，等大部分 Miner 完成后再切换
    """
    start: int = 0  # 起始 task_id (包含)
    end: int = 0    # 结束 task_id (不包含)
    
    @property
    def length(self) -> int:
        """范围内的 task 数量"""
        return max(0, self.end - self.start)
    
    def contains(self, task_id: int) -> bool:
        """检查 task_id 是否在范围内"""
        return self.start <= task_id < self.end
    
    def get_all_task_ids(self) -> List[int]:
        """获取范围内所有 task_id"""
        return list(range(self.start, self.end))
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "start": self.start,
            "end": self.end,
            "length": self.length
        }


@dataclass
class CalculationReport:
    """权重计算报告"""
    # Miner 统计
    total_miners: int = 0
    valid_miners: int = 0
    invalid_miners: int = 0
    incomplete_miners: List[Dict] = field(default_factory=list)
    
    # 环境比较结果
    env_comparisons: Dict[str, Any] = field(default_factory=dict)
    
    # 最终权重
    final_weights: Dict[str, float] = field(default_factory=dict)
    
    # 计算时间
    calculation_time_ms: int = 0
    
    # Task ID 范围 (用于记录本次计算使用的范围)
    task_id_ranges: Dict[str, TaskIdRange] = field(default_factory=dict)
    
    # 计时
    _start_time: float = field(default_factory=time.time)
    
    def end_timing(self) -> int:
        """结束计时，返回毫秒数"""
        return int((time.time() - self._start_time) * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于 JSON 序列化"""
        return {
            "total_miners": self.total_miners,
            "valid_miners": self.valid_miners,
            "invalid_miners": self.invalid_miners,
            "incomplete_miners": self.incomplete_miners,
            "env_comparisons": self.env_comparisons,
            "final_weights": self.final_weights,
            "calculation_time_ms": self.calculation_time_ms,
            "task_id_ranges": {
                env: range_obj.to_dict() 
                for env, range_obj in self.task_id_ranges.items()
            }
        }


@dataclass
class MinerInfo:
    """从 API 获取的 Miner 信息"""
    hotkey: str
    uid: int
    model_revision: str
    block_number: int  # 提交模型时的区块号
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MinerInfo":
        return cls(
            hotkey=data["hotkey"],
            uid=data["uid"],
            model_revision=data.get("model_revision", data.get("revision", "main")),
            block_number=data.get("block_number", 0)
        )


@dataclass
class SampleStatistics:
    """采样统计信息"""
    total_samples: int = 0
    average_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    success_rate: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SampleStatistics":
        if not data:
            return cls()
        return cls(
            total_samples=data.get("total_samples", 0),
            average_score=data.get("average_score", 0.0),
            min_score=data.get("min_score", 0.0),
            max_score=data.get("max_score", 0.0),
            success_rate=data.get("success_rate", 0.0)
        )


@dataclass 
class CompletionStatus:
    """采样完成状态"""
    is_complete: bool = False
    total_required: int = 0
    total_completed: int = 0
    missing_task_ids: List[int] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CompletionStatus":
        if not data:
            return cls()
        return cls(
            is_complete=data.get("is_complete", False),
            total_required=data.get("total_required", 0),
            total_completed=data.get("total_completed", 0),
            missing_task_ids=data.get("missing_task_ids", [])
        )
    
    @property
    def completion_percentage(self) -> float:
        """完成百分比"""
        if self.total_required == 0:
            return 0.0
        return (self.total_completed / self.total_required) * 100