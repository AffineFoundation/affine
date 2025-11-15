"""
Dynamic Threshold Algorithm for Scorer

核心思想: 后来者必须显著超过先来者才能获得权重
阈值随着分数接近 1.0 而减小，避免分数接近满分时无法超越
"""

import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class ThresholdConfig:
    """阈值配置"""
    base_threshold: float = 0.05  # 基础阈值 5%
    decay_alpha: float = 1.0      # 指数衰减参数
    log_beta: float = 2.0         # 对数参数


class DynamicThreshold:
    """
    动态阈值计算器
    
    核心问题: 后来者必须超过先来者一定比例才算"超过"，但这个比例不能是固定的。
    
    原因: 当分数接近 1.0 时，超越变得极其困难。例如：
    - 先来者分数 0.50，固定 5% 阈值意味着后来者需要 0.525
    - 先来者分数 0.95，固定 5% 阈值意味着后来者需要 0.9975 (几乎不可能)
    
    解决方案: 动态阈值，考虑剩余空间
    """
    
    def __init__(self, config: ThresholdConfig = None):
        """
        初始化动态阈值计算器
        
        Args:
            config: 阈值配置，如果为 None 则使用默认配置
        """
        self.config = config or ThresholdConfig()
    
    def calculate_required_score(self, current_score: float) -> float:
        """
        计算后来者需要达到的最小分数
        
        使用指数衰减算法:
        threshold = base_threshold * (1 - current_score)^alpha
        
        这种方法的优势:
        - 分数低时阈值高，容易超越
        - 分数高时阈值低，避免无法超越
        - 通过 alpha 参数控制衰减速度
        
        Args:
            current_score: 先来者的分数 (0.0 到 1.0)
        
        Returns:
            后来者需要超过的分数 (0.0 到 1.0)
        """
        if current_score < 0.0 or current_score > 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {current_score}")
        
        # 指数衰减: 阈值随分数指数减少
        # threshold = base_threshold * (1 - current_score)^alpha
        remaining_space = 1.0 - current_score
        threshold = self.config.base_threshold * math.pow(
            remaining_space,
            self.config.decay_alpha
        )
        return min(current_score + threshold, 1.0)
    
    def is_significant_improvement(self, new_score: float, old_score: float) -> bool:
        """
        判断新分数是否显著超过旧分数
        
        Args:
            new_score: 后来者分数 (0.0 到 1.0)
            old_score: 先来者分数 (0.0 到 1.0)
        
        Returns:
            True: 显著超过
            False: 未显著超过
        """
        required_score = self.calculate_required_score(old_score)
        return new_score >= required_score
    
    def get_improvement_percentage(self, new_score: float, old_score: float) -> float:
        """
        计算改进百分比
        
        Args:
            new_score: 后来者分数
            old_score: 先来者分数
        
        Returns:
            改进百分比 (可能为负数)
        """
        required_score = self.calculate_required_score(old_score)
        threshold_diff = required_score - old_score
        
        if threshold_diff <= 0:
            return float('inf')
        
        actual_improvement = new_score - old_score
        return (actual_improvement / threshold_diff) * 100
    
    def get_required_improvement_details(self, current_score: float) -> dict:
        """
        获取超越要求的详细信息
        
        Args:
            current_score: 当前分数
        
        Returns:
            详细信息字典
        """
        required = self.calculate_required_score(current_score)
        absolute_diff = required - current_score
        
        if current_score > 0:
            relative_diff = (absolute_diff / current_score) * 100
        else:
            relative_diff = float('inf')
        
        return {
            "current_score": current_score,
            "required_score": required,
            "absolute_improvement": absolute_diff,
            "relative_improvement_pct": relative_diff,
            "algorithm": "exponential_decay",
            "config": {
                "base_threshold": self.config.base_threshold,
                "decay_alpha": self.config.decay_alpha,
                "log_beta": self.config.log_beta,
            }
        }

def demonstrate_thresholds():
    """
    演示指数衰减阈值算法的效果
    """
    config = ThresholdConfig(base_threshold=0.05, decay_alpha=1.0)
    dt = DynamicThreshold(config)
    
    print("指数衰减动态阈值算法 (base=5%, alpha=1.0)")
    print("=" * 60)
    print("先来者分数 | 要求分数 | 绝对提升 | 相对提升")
    print("-" * 60)
    
    for score in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
        required = dt.calculate_required_score(score)
        abs_diff = required - score
        rel_diff = (abs_diff / score) * 100 if score > 0 else 0
        
        print(f"{score:.2f}       | {required:.4f}   | {abs_diff:.4f}   | {rel_diff:.2f}%")
    
    print("\n说明:")
    print("- 分数低时要求提升多，鼓励创新")
    print("- 分数高时要求提升少，避免无法超越")
    print("- 公式: required = score + 0.05 * (1 - score)^1.0")


def demonstrate_comparison():
    """
    演示 Miner 比较场景
    """
    config = ThresholdConfig(base_threshold=0.05, decay_alpha=1.0)
    dt = DynamicThreshold(config)
    
    print("\n场景: Miner A 分数 0.80，Miner B 分数 0.84")
    print("-" * 50)
    
    old_score = 0.80
    new_score = 0.84
    
    required = dt.calculate_required_score(old_score)
    is_significant = dt.is_significant_improvement(new_score, old_score)
    
    print(f"先来者 (A) 分数: {old_score}")
    print(f"后来者 (B) 分数: {new_score}")
    print(f"要求最低分数: {required:.4f}")
    print(f"是否显著超过: {is_significant}")
    
    if is_significant:
        print("✅ Miner B 显著超过 Miner A，可以获得权重")
    else:
        print("❌ Miner B 未显著超过 Miner A，不参与权重分配")


if __name__ == "__main__":
    demonstrate_thresholds()
    demonstrate_comparison()


"""
输出示例:

指数衰减动态阈值算法 (base=5%, alpha=1.0)
============================================================
先来者分数 | 要求分数 | 绝对提升 | 相对提升
------------------------------------------------------------
0.30       | 0.3350   | 0.0350   | 11.67%
0.50       | 0.5250   | 0.0250   | 5.00%
0.70       | 0.7150   | 0.0150   | 2.14%
0.80       | 0.8100   | 0.0100   | 1.25%
0.90       | 0.9050   | 0.0050   | 0.56%
0.95       | 0.9525   | 0.0025   | 0.26%
0.98       | 0.9810   | 0.0010   | 0.10%

说明:
- 分数低时要求提升多，鼓励创新
- 分数高时要求提升少，避免无法超越
- 公式: required = score + 0.05 * (1 - score)^1.0

场景: Miner A 分数 0.80，Miner B 分数 0.84
--------------------------------------------------
先来者 (A) 分数: 0.8
后来者 (B) 分数: 0.84
要求最低分数: 0.8100
是否显著超过: True
✅ Miner B 显著超过 Miner A，可以获得权重
"""