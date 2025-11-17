"""
Score Calculator - Weight Calculation Logic

Reuses existing sampling.py logic to calculate miner weights from all historical data.
"""

import time
import traceback
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict

from affine.sampling import SamplingOrchestrator, MinerSampler, SamplingConfig
from affine.core.setup import logger
from affine.core.http_client import AsyncHTTPClient


class ScoreCalculator:
    """Calculates miner weights from sampling data using existing sampling logic."""
    
    def __init__(self, api_base_url: str):
        """Initialize score calculator.
        
        Args:
            api_base_url: API server URL
        """
        self.api_base_url = api_base_url
        self.http_client = AsyncHTTPClient(timeout=60)
        
        # Reuse existing sampling logic
        self.orchestrator = SamplingOrchestrator()
        self.sampler = MinerSampler()
    
    async def fetch_all_samples(
        self,
        environments: Optional[List[str]] = None
    ) -> List[Dict]:
        """Fetch all sampling results from API.
        
        Args:
            environments: List of environments to fetch (None = all)
        
        Returns:
            List of sample dictionaries
        """
        try:
            url = f"{self.api_base_url}/api/v1/samples"
            params = {}
            
            if environments:
                params["env"] = ",".join(environments)
            
            response = await self.http_client.get(url, params=params)
            
            if response and isinstance(response, list):
                logger.info(f"[ScoreCalculator] Fetched {len(response)} samples")
                return response
            
            logger.warning("[ScoreCalculator] No samples fetched")
            return []
        
        except Exception as e:
            logger.error(f"[ScoreCalculator] Error fetching samples: {e}")
            traceback.print_exc()
            return []
    
    async def fetch_active_miners(self) -> Dict[int, str]:
        """Fetch active miners from metagraph.
        
        Returns:
            Dict mapping uid -> hotkey
        """
        try:
            url = f"{self.api_base_url}/api/v1/miners/active"
            response = await self.http_client.get(url)
            
            if response and isinstance(response, list):
                uid_to_hotkey = {
                    miner["uid"]: miner["hotkey"]
                    for miner in response
                }
                logger.info(f"[ScoreCalculator] Fetched {len(uid_to_hotkey)} active miners")
                return uid_to_hotkey
            
            logger.warning("[ScoreCalculator] No active miners fetched")
            return {}
        
        except Exception as e:
            logger.error(f"[ScoreCalculator] Error fetching miners: {e}")
            traceback.print_exc()
            return {}
    
    def _convert_samples_to_results(
        self,
        samples: List[Dict],
        uid_to_hotkey: Dict[int, str]
    ) -> List[Any]:
        """Convert API sample dictionaries to result objects for sampling.py.
        
        Args:
            samples: List of sample dictionaries from API
            uid_to_hotkey: Mapping of uid -> hotkey
        
        Returns:
            List of result objects compatible with sampling.py
        """
        from types import SimpleNamespace
        
        results = []
        
        for sample in samples:
            # Create nested namespace objects to match sampling.py expectations
            result = SimpleNamespace(
                miner=SimpleNamespace(
                    hotkey=sample.get("miner_hotkey"),
                    uid=sample.get("miner_uid"),
                    model=sample.get("model", ""),
                    revision=sample.get("revision", "main"),
                    block=sample.get("block", 0),
                ),
                challenge=SimpleNamespace(
                    env=sample.get("env", ""),
                    task_id=sample.get("task_id", 0),
                ),
                evaluation=SimpleNamespace(
                    score=sample.get("score", 0.0),
                    success=sample.get("success", False),
                ),
            )
            
            results.append(result)
        
        return results
    
    async def calculate_scores(
        self,
        environments: Tuple[str, ...],
        base_hotkey: str,
        burn: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Calculate miner weights from all historical samples.
        
        Args:
            environments: Tuple of environment names
            base_hotkey: Base validator hotkey (for burn mechanism)
            burn: Burn percentage (0.0-1.0)
        
        Returns:
            Tuple of (weights, details)
            where weights = {hotkey: weight}
            and details = {
                "eligible": List[str],
                "layer_points": {hotkey: {layer: points}},
                "env_winners": {env: hotkey},
                "confidence_intervals": {hotkey: {env: (lower, upper)}},
                "comprehensive_scores": {hotkey: score},
                "total_samples": int,
                "calculation_time": float,
            }
        """
        start_time = time.time()
        
        try:
            # Fetch active miners
            uid_to_hotkey = await self.fetch_active_miners()
            
            if not uid_to_hotkey:
                logger.warning("[ScoreCalculator] No active miners, returning empty weights")
                return {}, {
                    "eligible": [],
                    "total_samples": 0,
                    "calculation_time": time.time() - start_time,
                }
            
            meta_hotkeys = list(uid_to_hotkey.values())
            
            # Fetch all samples
            samples = await self.fetch_all_samples(list(environments))
            
            if not samples:
                logger.warning("[ScoreCalculator] No samples available, returning empty weights")
                return {}, {
                    "eligible": [],
                    "total_samples": 0,
                    "calculation_time": time.time() - start_time,
                }
            
            # Convert samples to result objects
            results = self._convert_samples_to_results(samples, uid_to_hotkey)
            
            # Process sample data using existing logic
            cnt, succ, prev, v_id, first_block, stats = self.orchestrator.process_sample_data(
                results=results,
                meta_hotkeys=meta_hotkeys,
                envs=environments,
                base_hk=base_hotkey,
            )
            
            # Determine queryable hotkeys (all active miners for now)
            queryable_hks = set(meta_hotkeys)
            
            # Calculate eligibility
            eligible_set, required = self.sampler.calculate_eligibility(
                cnt=cnt,
                active_hks=meta_hotkeys,
                queryable_hks=queryable_hks,
                envs=environments,
            )
            
            if not eligible_set:
                logger.warning("[ScoreCalculator] No eligible miners")
                return {}, {
                    "eligible": [],
                    "total_samples": len(samples),
                    "calculation_time": time.time() - start_time,
                }
            
            logger.info(
                f"[ScoreCalculator] Eligible miners: {len(eligible_set)} / {len(meta_hotkeys)}"
            )
            
            # Pre-calculate confidence intervals for all miners and environments
            confidence_intervals = {}
            for hk in eligible_set:
                confidence_intervals[hk] = {}
                for env in environments:
                    env_stats = stats.get(hk, {}).get(env, {})
                    samples_count = env_stats.get("samples", 0)
                    total_score = env_stats.get("total_score", 0.0)
                    
                    ci = self.sampler.challenge_algo.beta_confidence_interval(
                        total_score, samples_count
                    )
                    confidence_intervals[hk][env] = ci
            
            # Calculate combinatoric scores using existing logic
            scores, layer_points, env_winners = self.sampler.calculate_combinatoric_scores(
                envs=environments,
                pool=eligible_set,
                stats=stats,
                confidence_intervals=confidence_intervals,
            )
            
            # Calculate comprehensive scores for each miner
            comprehensive_scores = {}
            for hk in eligible_set:
                comprehensive_scores[hk] = self.sampler.calculate_comprehensive_score(
                    hotkey=hk,
                    env_subset=environments,
                    stats=stats,
                )
            
            # Calculate normalized weights
            weights, updated_eligible = self.orchestrator.calculate_weights(
                eligible=eligible_set,
                scores=scores,
                burn=burn,
                base_hotkey=base_hotkey,
            )
            
            calculation_time = time.time() - start_time
            
            logger.info(
                f"[ScoreCalculator] Calculated weights for {len(weights)} miners "
                f"in {calculation_time:.2f}s"
            )
            
            # Build detailed result
            details = {
                "eligible": sorted(updated_eligible),
                "layer_points": dict(layer_points),
                "env_winners": env_winners,
                "confidence_intervals": confidence_intervals,
                "comprehensive_scores": comprehensive_scores,
                "total_samples": len(samples),
                "calculation_time": calculation_time,
                "required_samples_per_env": required,
            }
            
            return weights, details
        
        except Exception as e:
            logger.error(f"[ScoreCalculator] Error calculating scores: {e}")
            traceback.print_exc()
            
            calculation_time = time.time() - start_time
            
            return {}, {
                "eligible": [],
                "error": str(e),
                "calculation_time": calculation_time,
            }