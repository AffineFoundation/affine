"""
Score Calculator V2 - Weight Calculation with Dynamic Threshold

基于新设计的分数计算器:
- 使用指数衰减动态阈值算法
- 支持 Task ID 范围（便于数据集扩展过渡）
- 后来者必须显著超过先来者
"""

import logging
import traceback
from typing import Dict, List, Any, Optional

from .api_client import ScorerAPIClient
from .comparator import MinerComparator
from .threshold import DynamicThreshold, ThresholdConfig
from .models import (
    MinerScore, 
    MinerInfo, 
    CalculationReport, 
    TaskIdRange
)
from .config import ScorerConfig

logger = logging.getLogger(__name__)


class ScoreCalculatorV2:
    """
    分数计算器核心逻辑 V2
    
    完整流程:
    1. 获取所有活跃 Miner
    2. 对每个 Miner，获取所有环境的采样结果（支持 Task ID 范围）
    3. 检查有效性 (是否完成所有采样)
    4. 计算每个环境的平均分
    5. 应用动态阈值比较
    6. 分配最终权重
    """
    
    def __init__(self, config: ScorerConfig):
        """
        初始化计算器
        
        Args:
            config: Scorer 配置
        """
        self.config = config
        self.api_client = ScorerAPIClient(
            config.api_base_url, 
            config.api_timeout_seconds
        )
        
        # 动态阈值配置
        threshold_config = ThresholdConfig(
            base_threshold=config.base_threshold,
            decay_alpha=config.decay_alpha
        )
        self.threshold = DynamicThreshold(threshold_config)
        self.comparator = MinerComparator(self.threshold)
    
    async def get_task_id_range_for_env(self, env: str) -> TaskIdRange:
        """
        获取环境的 Task ID 范围
        
        优先使用配置覆盖，否则从 API 获取
        
        Args:
            env: 环境名称
        
        Returns:
            TaskIdRange
        """
        # 优先使用覆盖配置
        if self.config.has_task_id_range_override(env):
            return self.config.get_task_id_range(env)
        
        # 从 API 获取
        try:
            dataset_info = await self.api_client.get_dataset_info(env)
            return TaskIdRange(
                start=dataset_info.get("task_id_start", 0),
                end=dataset_info.get("task_id_end", dataset_info.get("dataset_length", 0))
            )
        except Exception as e:
            logger.error(f"Failed to get dataset info for {env}: {e}")
            raise
    
    async def calculate_weights(
        self,
        environments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        计算所有 Miner 的权重
        
        Args:
            environments: 环境列表，None 表示使用配置中的默认值
        
        Returns:
            {
                "weights": {"hotkey": weight, ...},
                "report": CalculationReport
            }
        """
        if environments is None:
            environments = self.config.default_environments
        
        report = CalculationReport()
        
        try:
            # 1. 获取所有环境的 Task ID 范围
            for env in environments:
                report.task_id_ranges[env] = await self.get_task_id_range_for_env(env)
                logger.info(f"Task ID range for {env}: {report.task_id_ranges[env].to_dict()}")
            
            # 2. 获取所有活跃 Miner
            active_miners = await self.api_client.get_active_miners()
            report.total_miners = len(active_miners)
            
            if not active_miners:
                logger.warning("No active miners found")
                report.calculation_time_ms = report.end_timing()
                return {
                    "weights": {},
                    "report": report
                }
            
            # 3. 对每个 Miner 收集采样数据
            miner_data = {}  # hotkey -> {env -> MinerScore}
            
            for miner in active_miners:
                miner_scores = {}
                is_valid = True
                
                for env in environments:
                    task_id_range = report.task_id_ranges[env]
                    
                    # 获取采样结果
                    try:
                        samples_response = await self.api_client.get_miner_samples(
                            hotkey=miner.hotkey,
                            revision=miner.model_revision,
                            env=env,
                            dataset_length=task_id_range.length,
                            task_id_range=task_id_range,
                            dedupe_by_task_id=True
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get samples for {miner.hotkey}/{env}: {e}")
                        is_valid = False
                        report.incomplete_miners.append({
                            "hotkey": miner.hotkey,
                            "env": env,
                            "error": str(e)
                        })
                        break
                    
                    # 检查完整性 (新 API 格式)
                    is_complete = samples_response.get("is_complete", False)
                    if not is_complete:
                        is_valid = False
                        report.incomplete_miners.append({
                            "hotkey": miner.hotkey,
                            "env": env,
                            "total_required": samples_response.get("total_count", 0),
                            "total_completed": samples_response.get("completed_count", 0),
                            "missing_count": len(samples_response.get("missing_task_ids", [])),
                            "missing_task_ids": samples_response.get("missing_task_ids", [])[:10]  # 只记录前10个
                        })
                        break
                    
                    # 计算平均分 (从采样数据计算)
                    samples = samples_response.get("samples", [])
                    if samples:
                        scores = [s.get("score", 0.0) for s in samples]
                        average_score = sum(scores) / len(scores)
                    else:
                        average_score = 0.0
                    
                    miner_scores[env] = MinerScore(
                        hotkey=miner.hotkey,
                        uid=miner.uid,
                        model_revision=miner.model_revision,
                        env=env,
                        average_score=average_score,
                        block_number=miner.block_number
                    )
                
                if is_valid:
                    miner_data[miner.hotkey] = miner_scores
                    report.valid_miners += 1
                else:
                    report.invalid_miners += 1
            
            logger.info(
                f"Valid miners: {report.valid_miners}/{report.total_miners}, "
                f"Invalid: {report.invalid_miners}"
            )
            
            if not miner_data:
                logger.warning("No valid miners with complete samples")
                report.calculation_time_ms = report.end_timing()
                return {
                    "weights": {},
                    "report": report
                }
            
            # 4. 对每个环境进行比较
            env_weights = {}  # env -> {hotkey -> weight}
            
            for env in environments:
                # 收集该环境下所有有效 Miner 的分数
                env_miners = [
                    miner_scores[env] 
                    for miner_scores in miner_data.values()
                ]
                
                if not env_miners:
                    continue
                
                # 比较并标记
                compared_miners = self.comparator.compare_miners_for_env(env_miners)
                
                # 分配权重
                env_weights[env] = self.comparator.assign_weights(
                    compared_miners,
                    weight_method=self.config.weight_method
                )
                
                # 记录比较结果
                report.env_comparisons[env] = self.comparator.get_comparison_report(
                    compared_miners
                )
            
            # 5. 合并所有环境的权重
            final_weights = self._aggregate_env_weights(env_weights, environments)
            report.final_weights = final_weights
            report.calculation_time_ms = report.end_timing()
            
            logger.info(
                f"Calculated weights for {len(final_weights)} miners "
                f"in {report.calculation_time_ms}ms"
            )
            
            return {
                "weights": final_weights,
                "report": report
            }
        
        except Exception as e:
            logger.error(f"Error calculating weights: {e}")
            traceback.print_exc()
            
            report.calculation_time_ms = report.end_timing()
            
            return {
                "weights": {},
                "report": report
            }
    
    def _aggregate_env_weights(
        self,
        env_weights: Dict[str, Dict[str, float]],
        environments: List[str]
    ) -> Dict[str, float]:
        """
        聚合所有环境的权重
        
        方法: 对每个 Miner，取所有环境权重的平均值
        
        Args:
            env_weights: env -> {hotkey -> weight}
            environments: 环境列表
        
        Returns:
            hotkey -> final_weight
        """
        if not env_weights:
            return {}
        
        # 收集所有 hotkey
        all_hotkeys = set()
        for weights in env_weights.values():
            all_hotkeys.update(weights.keys())
        
        final_weights = {}
        
        for hotkey in all_hotkeys:
            # 收集该 Miner 在所有环境中的权重
            hotkey_weights = []
            for env in environments:
                if env in env_weights and hotkey in env_weights[env]:
                    hotkey_weights.append(env_weights[env][hotkey])
            
            if hotkey_weights:
                # 平均值
                final_weights[hotkey] = sum(hotkey_weights) / len(hotkey_weights)
        
        # 归一化
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            for hotkey in final_weights:
                final_weights[hotkey] /= total_weight
        
        return final_weights
    
    async def run_calculation(self) -> str:
        """
        执行一次完整计算并保存结果
        
        Returns:
            snapshot_id
        """
        # 计算权重
        result = await self.calculate_weights()
        
        if not result["weights"]:
            logger.warning("No weights calculated, skipping save")
            return "no_weights"
        
        # 保存到 API
        try:
            block_number = await self.api_client.get_current_block()
            
            snapshot_id = await self.api_client.save_weights(
                weights=result["weights"],
                calculation_details=result["report"].to_dict(),
                block_number=block_number
            )
            
            logger.info(f"Weight calculation completed, snapshot: {snapshot_id}")
            return snapshot_id
        
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
            raise
    
    async def close(self):
        """关闭资源"""
        await self.api_client.close()