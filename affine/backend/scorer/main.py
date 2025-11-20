"""
Scorer Service Main Entry

Score calculator service main entry point.
- Supports scheduled service mode
- Supports single execution mode (debugging)
- Supports command line parameter configuration
"""

import asyncio
import click
from typing import Optional

from affine.core.setup import setup_logging, logger
from .calculator_v2 import ScoreCalculatorV2
from .config import ScorerConfig


class ScorerService:
    """
    Scorer Service
    
    Runs weight calculation periodically.
    """
    
    def __init__(self, config: ScorerConfig):
        """
        Initialize service.
        
        Args:
            config: Scorer configuration
        """
        self.config = config
        self.calculator = ScoreCalculatorV2(config)
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start service"""
        self.running = True
        
        logger.info("Starting Scorer Service")
        logger.info(f"  API URL: {self.config.api_base_url}")
        logger.info(f"  Calculation interval: {self.config.calculation_interval_minutes} minutes")
        logger.info(f"  Base threshold: {self.config.base_threshold}")
        logger.info(f"  Decay alpha: {self.config.decay_alpha}")
        logger.info(f"  Environments: {self.config.default_environments}")
        
        try:
            # Execute immediately once
            await self._run_calculation()
            
            # Enter loop
            while self.running:
                # Wait for next calculation
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
        """Stop service"""
        self.running = False
        
        # Close resources
        await self.calculator.close()
        
        logger.info("Scorer Service stopped")
    
    async def _run_calculation(self):
        """Execute one calculation"""
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
        Single execution mode (debugging)
        
        Returns:
            Calculation result
        """
        try:
            result = await self.calculator.calculate_weights()
            return result
        finally:
            await self.calculator.close()


def print_result(result: dict):
    """Print calculation result"""
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
        # Sort by weight
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for hotkey, weight in sorted_weights[:10]:  # Only show top 10
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
    help='Base threshold percentage (e.g., 0.05 for 5%%)'
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
    '-v', '--verbosity',
    default='1',
    type=click.Choice(['0', '1', '2', '3']),
    help='Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE'
)
def main(api_url, interval, base_threshold, decay_alpha, weight_method, envs, once, verbosity):
    """
    Affine Scorer - Weight calculation service
    
    Calculates miner weights based on sampling results using dynamic threshold algorithm.
    """
    # Setup logging
    setup_logging(int(verbosity))
    
    # Parse environment list
    environments = [e.strip() for e in envs.split(',') if e.strip()]
    
    # Create configuration
    config = ScorerConfig(
        api_base_url=api_url,
        calculation_interval_minutes=interval,
        base_threshold=base_threshold,
        decay_alpha=decay_alpha,
        weight_method=weight_method,
        default_environments=environments
    )
    
    # Create service
    service = ScorerService(config)
    
    if once:
        # Single execution mode
        logger.info("Running in single calculation mode")
        result = asyncio.run(service.run_once())
        print_result(result)
    else:
        # Service mode
        logger.info("Running in service mode")
        try:
            asyncio.run(service.start())
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    main()