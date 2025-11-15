"""
Scorer Configuration

分数计算器配置，支持环境变量覆盖
"""

from pydantic_settings import BaseSettings
from typing import Dict
from .models import TaskIdRange


class ScorerConfig(BaseSettings):
    """Scorer 配置"""
    
    # API 配置
    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: int = 60
    
    # 计算间隔
    calculation_interval_minutes: int = 30
    
    # 动态阈值配置 (指数衰减算法)
    # threshold = base_threshold * (1 - current_score)^decay_alpha
    base_threshold: float = 0.05  # 基础阈值 5%
    decay_alpha: float = 1.0      # 指数衰减参数
    
    # 权重分配方法: score_proportional, rank_based, equal
    weight_method: str = "score_proportional"
    
    # 默认环境列表
    default_environments: list = [
        "affine:sat", 
        "affine:abd", 
        "affine:ded"
    ]
    
    # 每个环境的 Task ID 范围 (用于支持数据集扩展过渡)
    # 如果未指定，将从 API 获取当前数据集长度
    # 格式: {"affine:sat": {"start": 0, "end": 100}, ...}
    task_id_ranges_override: Dict[str, Dict[str, int]] = {}
    
    # 日志级别
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "AFFINE_SCORER_"
        extra = "ignore"
    
    def get_task_id_range(self, env: str) -> TaskIdRange:
        """
        获取指定环境的 Task ID 范围
        
        如果有覆盖配置则使用覆盖，否则返回空范围（需要从 API 获取）
        """
        if env in self.task_id_ranges_override:
            range_dict = self.task_id_ranges_override[env]
            return TaskIdRange(
                start=range_dict.get("start", 0),
                end=range_dict.get("end", 0)
            )
        return TaskIdRange(start=0, end=0)
    
    def has_task_id_range_override(self, env: str) -> bool:
        """检查是否有 Task ID 范围覆盖"""
        return env in self.task_id_ranges_override