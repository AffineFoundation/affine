"""
Miner Comparator - Apply dynamic threshold to compare miners

核心逻辑:
1. 按提交时间排序 (block_number)
2. 后来者必须显著超过所有先来者
3. 应用动态阈值
"""

from typing import List, Dict, Optional
from .threshold import DynamicThreshold, ThresholdConfig
from .models import MinerScore


class MinerComparator:
    """
    Miner 比较器
    
    核心职责:
    - 按提交时间排序 Miner
    - 应用动态阈值比较
    - 确定哪些 Miner 是"优越的"
    - 分配权重
    """
    
    def __init__(self, threshold: DynamicThreshold = None):
        """
        初始化比较器
        
        Args:
            threshold: 动态阈值计算器，如果为 None 则使用默认配置
        """
        if threshold is None:
            config = ThresholdConfig()
            self.threshold = DynamicThreshold(config)
        else:
            self.threshold = threshold
    
    def compare_miners_for_env(self, miners: List[MinerScore]) -> List[MinerScore]:
        """
        在单个环境中比较所有 Miner
        
        使用指数衰减动态阈值算法，后来者必须显著超过先来者
        
        Args:
            miners: Miner 分数列表 (包含 block_number 和 average_score)
        
        Returns:
            更新后的 Miner 列表，包含 is_superior 标记
        """
        if not miners:
            return []
        
        if len(miners) == 1:
            # 只有一个 Miner，默认为优越
            miners[0].is_superior = True
            miners[0].superiority_reason = "single_miner"
            return miners
        
        # 1. 按提交时间排序 (block_number 越小越早)
        sorted_miners = sorted(miners, key=lambda m: (m.block_number, m.hotkey))
        
        # 2. 第一个永远有效 (先来者优势)
        sorted_miners[0].is_superior = True
        sorted_miners[0].superiority_reason = "first_submitter"
        
        # 3. 对于后来者，检查是否显著超过所有优越的先来者
        for i in range(1, len(sorted_miners)):
            current_miner = sorted_miners[i]
            
            # 找到目前最高分的优越先来者
            superior_prior_miners = [
                m for m in sorted_miners[:i] 
                if m.is_superior
            ]
            
            if not superior_prior_miners:
                # 没有优越的先来者（不应该发生，因为第一个永远优越）
                current_miner.is_superior = True
                current_miner.superiority_reason = "no_superior_prior"
                continue
            
            best_prior_score = max(m.average_score for m in superior_prior_miners)
            
            # 检查是否显著超过
            if self.threshold.is_significant_improvement(
                current_miner.average_score,
                best_prior_score
            ):
                current_miner.is_superior = True
                current_miner.superiority_reason = (
                    f"exceeded_score_{best_prior_score:.4f}_with_{current_miner.average_score:.4f}"
                )
            else:
                current_miner.is_superior = False
                required = self.threshold.calculate_required_score(best_prior_score)
                current_miner.superiority_reason = (
                    f"not_exceeded_required_{required:.4f}_got_{current_miner.average_score:.4f}"
                )
        
        return sorted_miners
    
    def assign_weights(
        self,
        miners: List[MinerScore],
        weight_method: str = "score_proportional"
    ) -> Dict[str, float]:
        """
        为 Miner 分配权重
        
        Args:
            miners: 已标记 is_superior 的 Miner 列表
            weight_method: 权重分配方法
                - "score_proportional": 按分数比例分配 (推荐)
                - "rank_based": 按排名分配
                - "equal": 平均分配
        
        Returns:
            hotkey -> weight 映射 (归一化后的权重)
        """
        # 只有 is_superior 的 Miner 才有权重
        superior_miners = [m for m in miners if m.is_superior]
        
        if not superior_miners:
            return {}
        
        weights = {}
        
        if weight_method == "score_proportional":
            # 按分数比例分配
            total_score = sum(m.average_score for m in superior_miners)
            
            if total_score > 0:
                for miner in superior_miners:
                    weights[miner.hotkey] = miner.average_score / total_score
            else:
                # 所有分数为 0，平均分配
                weight_per_miner = 1.0 / len(superior_miners)
                for miner in superior_miners:
                    weights[miner.hotkey] = weight_per_miner
        
        elif weight_method == "rank_based":
            # 按排名分配 (分数越高权重越大)
            ranked = sorted(superior_miners, key=lambda m: m.average_score, reverse=True)
            
            # 使用逆排名作为权重 (第1名 = n分，第2名 = n-1分，...)
            n = len(ranked)
            total_rank = sum(range(1, n + 1))  # 1 + 2 + ... + n
            
            for i, miner in enumerate(ranked):
                rank_weight = (n - i) / total_rank
                weights[miner.hotkey] = rank_weight
        
        elif weight_method == "equal":
            # 平均分配
            weight_per_miner = 1.0 / len(superior_miners)
            for miner in superior_miners:
                weights[miner.hotkey] = weight_per_miner
        
        else:
            raise ValueError(f"Unknown weight method: {weight_method}")
        
        return weights
    
    def get_comparison_report(self, miners: List[MinerScore]) -> Dict:
        """
        生成比较报告
        
        Args:
            miners: Miner 列表 (必须已经调用过 compare_miners_for_env)
        
        Returns:
            详细报告字典
        """
        if not miners:
            return {
                "total_miners": 0,
                "superior_miners": 0,
                "inferior_miners": 0,
                "details": []
            }
        
        # 按 block_number 排序
        sorted_miners = sorted(miners, key=lambda m: (m.block_number, m.hotkey))
        
        details = []
        for miner in sorted_miners:
            detail = {
                "hotkey": miner.hotkey,
                "uid": miner.uid,
                "model_revision": miner.model_revision,
                "average_score": miner.average_score,
                "block_number": miner.block_number,
                "is_superior": miner.is_superior,
                "reason": miner.superiority_reason
            }
            details.append(detail)
        
        superior_count = sum(1 for m in miners if m.is_superior)
        
        return {
            "total_miners": len(miners),
            "superior_miners": superior_count,
            "inferior_miners": len(miners) - superior_count,
            "threshold_algorithm": "exponential_decay",
            "threshold_config": {
                "base_threshold": self.threshold.config.base_threshold,
                "decay_alpha": self.threshold.config.decay_alpha,
                "log_beta": self.threshold.config.log_beta
            },
            "details": details
        }


def demonstrate_miner_comparison():
    """
    演示 Miner 比较流程
    """
    # 创建测试数据
    miners = [
        MinerScore(
            hotkey="miner_a",
            uid=1,
            model_revision="rev_a",
            env="test_env",
            average_score=0.75,
            block_number=1000,  # 最早提交
        ),
        MinerScore(
            hotkey="miner_b",
            uid=2,
            model_revision="rev_b",
            env="test_env",
            average_score=0.78,  # 略高于 A，但可能不够
            block_number=1100,
        ),
        MinerScore(
            hotkey="miner_c",
            uid=3,
            model_revision="rev_c",
            env="test_env",
            average_score=0.85,  # 明显高于 A
            block_number=1200,
        ),
        MinerScore(
            hotkey="miner_d",
            uid=4,
            model_revision="rev_d",
            env="test_env",
            average_score=0.80,  # 低于 C
            block_number=1300,
        ),
    ]
    
    config = ThresholdConfig(base_threshold=0.05, decay_alpha=1.0)
    threshold = DynamicThreshold(config)
    comparator = MinerComparator(threshold)
    
    print("=== Miner 比较演示 ===\n")
    print("输入 Miners:")
    for m in miners:
        print(f"  {m.hotkey}: score={m.average_score}, block={m.block_number}")
    
    print("\n比较结果:")
    result = comparator.compare_miners_for_env(miners)
    
    for m in result:
        status = "✅ 优越" if m.is_superior else "❌ 劣势"
        print(f"  {m.hotkey}: {status}")
        print(f"    原因: {m.superiority_reason}")
    
    print("\n权重分配:")
    weights = comparator.assign_weights(result, "score_proportional")
    for hotkey, weight in weights.items():
        print(f"  {hotkey}: {weight:.4f}")
    
    print("\n详细报告:")
    report = comparator.get_comparison_report(result)
    print(f"  总 Miners: {report['total_miners']}")
    print(f"  优越 Miners: {report['superior_miners']}")
    print(f"  劣势 Miners: {report['inferior_miners']}")


if __name__ == "__main__":
    demonstrate_miner_comparison()


"""
输出示例:

=== Miner 比较演示 ===

输入 Miners:
  miner_a: score=0.75, block=1000
  miner_b: score=0.78, block=1100
  miner_c: score=0.85, block=1200
  miner_d: score=0.80, block=1300

比较结果:
  miner_a: ✅ 优越
    原因: first_submitter
  miner_b: ❌ 劣势
    原因: not_exceeded_required_0.7625_got_0.7800
  miner_c: ✅ 优越
    原因: exceeded_score_0.7500_with_0.8500
  miner_d: ❌ 劣势
    原因: not_exceeded_required_0.8575_got_0.8000

权重分配:
  miner_a: 0.4688
  miner_c: 0.5312

详细报告:
  总 Miners: 4
  优越 Miners: 2
  劣势 Miners: 2
"""