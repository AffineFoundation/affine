"""
Scorer Service Main Entry

分数计算器服务主入口
- 支持定时服务模式
- 支持单次执行模式（调试）
- 支持命令行参数配置
"""

import asyncio
import logging
import click
from typing import Optional

from .calculator_v2 import ScoreCalculatorV2
from .config import ScorerConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScorerService:
    """
    计分器服务
    
    定时运行权重计算
    """
    
    def __init__(self, config: ScorerConfig):
        """
        初始化服务
        
        Args:
            config: Scorer 配置
        """
        self.config = config
        self.calculator = ScoreCalculatorV2(config)
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动服务"""
        self.running = True
        
        logger.info("Starting Scorer Service")
        logger.info(f"  API URL: {self.config.api_base_url}")
        logger.info(f"  Calculation interval: {self.config.calculation_interval_minutes} minutes")
        logger.info(f"  Base threshold: {self.config.base_threshold}")
        logger.info(f"  Decay alpha: {self.config.decay_alpha}")
        logger.info(f"  Environments: {self.config.default_environments}")
        
        try:
            # 立即执行一次
            await self._run_calculation()
            
            # 进入循环
            while self.running:
                # 等待下次计算
                await asyncio.sleep(self.config.calculation_interval_minutes * 60)
                
                if self.running:
                    await self._run_calculation()
        
        except asyncio.CancelledError:
            logger.info("Service cancelled")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            await self.stop()
    
    async def stop(self):
        """停止服务"""
        self.running = False
        
        # 关闭资源
        await self.calculator.close()
        
        logger.info("Scorer Service stopped")
    
    async def _run_calculation(self):
        """执行一次计算"""
        logger.info("=" * 60)
        logger.info("Starting weight calculation")
        
        try:
            snapshot_id = await self.calculator.run_calculation()
            logger.info(f"Weight calculation completed, snapshot: {snapshot_id}")
        except Exception as e:
            logger.error(f"Weight calculation failed: {e}")
        
        logger.info("=" * 60)
    
    async def run_once(self) -> dict:
        """
        单次执行模式 (调试)
        
        Returns:
            计算结果
        """
        try:
            result = await self.calculator.calculate_weights()
            return result
        finally:
            await self.calculator.close()


def print_result(result: dict):
    """打印计算结果"""
    report = result["report"]
    weights = result["weights"]
    
    print("\n" + "=" * 60)
    print("Weight Calculation Result")
    print("=" * 60)
    
    print(f"\nMiners:")
    print(f"  Total: {report.total_miners}")
    print(f"  Valid: {report.valid_miners}")
    print(f"  Invalid: {report.invalid_miners}")
    
    if report.incomplete_miners:
        print(f"\nIncomplete Miners (first 5):")
        for i, info in enumerate(report.incomplete_miners[:5]):
            print(f"  {i+1}. {info.get('hotkey', 'unknown')[:16]}... - {info.get('env', 'unknown')}")
            if "missing_count" in info:
                print(f"     Missing: {info['missing_count']} tasks")
    
    print(f"\nTask ID Ranges:")
    for env, range_obj in report.task_id_ranges.items():
        print(f"  {env}: {range_obj.to_dict()}")
    
    if report.env_comparisons:
        print(f"\nEnvironment Comparisons:")
        for env, comparison in report.env_comparisons.items():
            print(f"  {env}:")
            print(f"    Total: {comparison['total_miners']}")
            print(f"    Superior: {comparison['superior_miners']}")
            print(f"    Inferior: {comparison['inferior_miners']}")
    
    if weights:
        print(f"\nFinal Weights:")
        # 按权重排序
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for hotkey, weight in sorted_weights[:10]:  # 只显示前10
            print(f"  {hotkey[:16]}... : {weight:.6f}")
        if len(sorted_weights) > 10:
            print(f"  ... and {len(sorted_weights) - 10} more")
    else:
        print("\nNo weights calculated")
    
    print(f"\nCalculation Time: {report.calculation_time_ms}ms")
    print("=" * 60)


@click.command()
@click.option(
    '--api-url', 
    default='http://localhost:8000', 
    help='API base URL'
)
@click.option(
    '--interval', 
    default=30, 
    type=int, 
    help='Calculation interval in minutes'
)
@click.option(
    '--base-threshold', 
    default=0.05, 
    type=float, 
    help='Base threshold percentage (e.g., 0.05 for 5%)'
)
@click.option(
    '--decay-alpha', 
    default=1.0, 
    type=float, 
    help='Exponential decay alpha parameter'
)
@click.option(
    '--weight-method', 
    default='score_proportional',
    type=click.Choice(['score_proportional', 'rank_based', 'equal']),
    help='Weight distribution method'
)
@click.option(
    '--envs',
    default='affine:sat,affine:abd,affine:ded',
    help='Comma-separated list of environments'
)
@click.option(
    '--once', 
    is_flag=True, 
    help='Run one calculation and exit (debug mode)'
)
@click.option(
    '--log-level',
    default='INFO',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    help='Logging level'
)
def main(api_url, interval, base_threshold, decay_alpha, weight_method, envs, once, log_level):
    """
    Affine Scorer - Weight calculation service
    
    Calculates miner weights based on sampling results using dynamic threshold algorithm.
    """
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # 解析环境列表
    environments = [e.strip() for e in envs.split(',') if e.strip()]
    
    # 创建配置
    config = ScorerConfig(
        api_base_url=api_url,
        calculation_interval_minutes=interval,
        base_threshold=base_threshold,
        decay_alpha=decay_alpha,
        weight_method=weight_method,
        default_environments=environments
    )
    
    # 创建服务
    service = ScorerService(config)
    
    if once:
        # 单次执行模式
        logger.info("Running in single calculation mode")
        result = asyncio.run(service.run_once())
        print_result(result)
    else:
        # 服务模式
        logger.info("Running in service mode")
        try:
            asyncio.run(service.start())
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    main()