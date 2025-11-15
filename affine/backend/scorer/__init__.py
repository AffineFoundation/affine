"""
Scorer - Weight Calculation Service

分数计算器服务，负责:
- 从 API 层获取采样数据
- 应用动态阈值算法比较 Miner
- 计算权重分配
- 保存权重快照

核心算法:
- 指数衰减动态阈值: threshold = base_threshold * (1 - score)^alpha
- 后来者必须显著超过先来者才能获得权重
- 支持 Task ID 范围查询（便于数据集扩展过渡）

使用示例:
```python
from affine.backend.scorer import ScorerService, ScorerConfig

config = ScorerConfig(
    api_base_url="http://localhost:8000",
    base_threshold=0.05,
    decay_alpha=1.0
)
service = ScorerService(config)

# 服务模式
await service.start()

# 或单次执行
result = await service.run_once()
```

命令行使用:
```bash
# 服务模式
python -m affine.backend.scorer.main --api-url http://localhost:8000

# 单次执行
python -m affine.backend.scorer.main --once
```
"""

from .config import ScorerConfig
from .threshold import DynamicThreshold, ThresholdConfig
from .comparator import MinerComparator
from .models import (
    MinerScore,
    MinerInfo,
    TaskIdRange,
    CalculationReport,
    SampleStatistics,
    CompletionStatus
)
from .api_client import ScorerAPIClient
from .calculator_v2 import ScoreCalculatorV2
from .main import ScorerService

__all__ = [
    # Main Service
    "ScorerService",
    
    # Calculator
    "ScoreCalculatorV2",
    
    # Configuration
    "ScorerConfig",
    
    # Core Algorithm
    "DynamicThreshold",
    "ThresholdConfig",
    "MinerComparator",
    
    # Data Models
    "MinerScore",
    "MinerInfo",
    "TaskIdRange",
    "CalculationReport",
    "SampleStatistics",
    "CompletionStatus",
    
    # API Client
    "ScorerAPIClient",
]