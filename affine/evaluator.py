#!/usr/bin/env python3
"""
GRPO (Group Relative Performance Optimization) Evaluator

This module implements a Sybil-proof evaluation system using:
1. Group Relative Performance Optimization (GRPO)
2. Per-model aggregation to collapse Sybil clones
3. min(score_domain1, score_domain2) to punish imbalance
4. Winner-take-all rewards

Key formulas:
- score_domain[i] = GRPO_domain[i] = S_domain[i] / mean(S_domain)
- final_score[i] = min(grpo_domain1[i], grpo_domain2[i])
- winner = argmax(final_score)
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pydantic import BaseModel

logger = logging.getLogger("affine.evaluator")

@dataclass
class MinerPerformance:
    """Performance data for a single miner across domains"""
    uid: int
    hotkey: str
    model: Optional[str] = None
    domain_scores: Dict[str, List[float]] = field(default_factory=dict)
    domain_averages: Dict[str, float] = field(default_factory=dict)
    grpo_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    sample_counts: Dict[str, int] = field(default_factory=dict)
    
    def add_score(self, domain: str, score: float):
        """Add a score for a specific domain"""
        if domain not in self.domain_scores:
            self.domain_scores[domain] = []
        self.domain_scores[domain].append(score)
        self.domain_averages[domain] = sum(self.domain_scores[domain]) / len(self.domain_scores[domain])
        self.sample_counts[domain] = len(self.domain_scores[domain])
    
    def update_grpo_score(self, domain: str, grpo_score: float):
        """Update GRPO score for a domain"""
        self.grpo_scores[domain] = grpo_score
    
    def update_final_score(self, required_domains: List[str]):
        """Update final score using min() across required domains"""
        if all(domain in self.grpo_scores for domain in required_domains):
            self.final_score = min(self.grpo_scores[domain] for domain in required_domains)
        else:
            self.final_score = 0.0

@dataclass 
class EvaluationRound:
    """Data for a single evaluation round"""
    round_id: str
    timestamp: float
    domain_group_averages: Dict[str, float] = field(default_factory=dict)
    miner_performances: Dict[int, MinerPerformance] = field(default_factory=dict)
    winner_uid: Optional[int] = None
    winner_model: Optional[str] = None
    winner_score: float = 0.0

class Evaluator:
    """
    GRPO-based evaluator implementing Sybil-proof scoring
    
    Features:
    - Group Relative Performance Optimization (GRPO)
    - Per-model aggregation (collapses Sybil clones)
    - Multi-domain scoring with min() penalty for imbalance
    - Winner-take-all rewards
    - Exponential moving average for smooth updates
    - Score history tracking
    - Penalty for domain skew
    - First-come priority: When multiple miners run the same model, rewards go to the oldest commitment
    """
    
    def __init__(self, 
                 required_domains: List[str] = ["SAT", "ABD"],
                 storage_path: str = "~/.affine/results/evaluator_scores.json",
                 ema_alpha: float = 0.1,
                 skew_penalty_weight: float = 0.1):
        """
        Initialize evaluator
        
        Args:
            required_domains: Domains that must be present for scoring
            storage_path: Path to store evaluation history
            ema_alpha: Exponential moving average smoothing factor
            skew_penalty_weight: Weight for domain skew penalty
        """
        self.required_domains = required_domains
        self.storage_path = os.path.expanduser(storage_path)
        self.ema_alpha = ema_alpha
        self.skew_penalty_weight = skew_penalty_weight
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Load existing data
        self.evaluation_history: List[EvaluationRound] = []
        self.miner_history: Dict[int, MinerPerformance] = {}
        self._load_history()
    
    def _load_history(self):
        """Load evaluation history from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # Load evaluation rounds
                for round_data in data.get('evaluation_history', []):
                    round_obj = EvaluationRound(
                        round_id=round_data['round_id'],
                        timestamp=round_data['timestamp'],
                        domain_group_averages=round_data.get('domain_group_averages', {}),
                        winner_uid=round_data.get('winner_uid'),
                        winner_model=round_data.get('winner_model'),
                        winner_score=round_data.get('winner_score', 0.0)
                    )
                    
                    # Reconstruct miner performances for this round
                    for uid_str, perf_data in round_data.get('miner_performances', {}).items():
                        uid = int(uid_str)
                        perf = MinerPerformance(
                            uid=uid,
                            hotkey=perf_data['hotkey'],
                            model=perf_data.get('model'),
                            domain_scores=perf_data.get('domain_scores', {}),
                            domain_averages=perf_data.get('domain_averages', {}),
                            grpo_scores=perf_data.get('grpo_scores', {}),
                            final_score=perf_data.get('final_score', 0.0),
                            sample_counts=perf_data.get('sample_counts', {})
                        )
                        round_obj.miner_performances[uid] = perf
                    
                    self.evaluation_history.append(round_obj)
                
                # Load cumulative miner history
                for uid_str, perf_data in data.get('miner_history', {}).items():
                    uid = int(uid_str)
                    perf = MinerPerformance(
                        uid=uid,
                        hotkey=perf_data['hotkey'],
                        model=perf_data.get('model'),
                        domain_scores=perf_data.get('domain_scores', {}),
                        domain_averages=perf_data.get('domain_averages', {}),
                        grpo_scores=perf_data.get('grpo_scores', {}),
                        final_score=perf_data.get('final_score', 0.0),
                        sample_counts=perf_data.get('sample_counts', {})
                    )
                    self.miner_history[uid] = perf
                
                logger.info(f"Loaded {len(self.evaluation_history)} evaluation rounds and {len(self.miner_history)} miner histories")
                
            except Exception as e:
                logger.warning(f"Failed to load evaluation history: {e}")
                self.evaluation_history = []
                self.miner_history = {}
    
    def _save_history(self):
        """Save evaluation history to disk"""
        try:
            # Prepare data for serialization
            data = {
                'evaluation_history': [],
                'miner_history': {}
            }
            
            # Serialize evaluation rounds
            for round_obj in self.evaluation_history:
                round_data = {
                    'round_id': round_obj.round_id,
                    'timestamp': round_obj.timestamp,
                    'domain_group_averages': round_obj.domain_group_averages,
                    'winner_uid': round_obj.winner_uid,
                    'winner_model': round_obj.winner_model,
                    'winner_score': round_obj.winner_score,
                    'miner_performances': {}
                }
                
                for uid, perf in round_obj.miner_performances.items():
                    round_data['miner_performances'][str(uid)] = {
                        'hotkey': perf.hotkey,
                        'model': perf.model,
                        'domain_scores': perf.domain_scores,
                        'domain_averages': perf.domain_averages,
                        'grpo_scores': perf.grpo_scores,
                        'final_score': perf.final_score,
                        'sample_counts': perf.sample_counts
                    }
                
                data['evaluation_history'].append(round_data)
            
            # Serialize miner history
            for uid, perf in self.miner_history.items():
                data['miner_history'][str(uid)] = {
                    'hotkey': perf.hotkey,
                    'model': perf.model,
                    'domain_scores': perf.domain_scores,
                    'domain_averages': perf.domain_averages,
                    'grpo_scores': perf.grpo_scores,
                    'final_score': perf.final_score,
                    'sample_counts': perf.sample_counts
                }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved evaluation history to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation history: {e}")
    
    def aggregate_by_model(self, results: List[Any]) -> Dict[str, List[Any]]:
        """
        Aggregate results by model hash to collapse Sybil clones.
        When multiple miners run the same model, the miner with the oldest 
        block commitment gets priority for rewards.
        
        Args:
            results: List of result objects with miner info
            
        Returns:
            Dict mapping model_hash -> list of results for that model
        """
        model_groups = defaultdict(list)
        
        for result in results:
            model = result.miner.model or "unknown"
            model_groups[model].append(result)
        
        return dict(model_groups)
    
    def calculate_grpo_scores(self, 
                            domain_results: Dict[str, List[Any]], 
                            model_groups: Dict[str, List[Any]]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Calculate GRPO scores for each domain
        
        Args:
            domain_results: Dict mapping domain -> list of results
            model_groups: Dict mapping model -> list of all results for that model
            
        Returns:
            Tuple of (domain_grpo_scores, domain_group_averages)
            domain_grpo_scores[domain][model] = GRPO score
        """
        domain_grpo_scores = {}
        domain_group_averages = {}
        
        for domain, results in domain_results.items():
            if not results:
                continue
                
            # Calculate raw scores by model (average across miners with same model)
            model_scores = {}
            for model, model_results in model_groups.items():
                # Get results for this model in this domain
                domain_model_results = [r for r in model_results if any(r.miner.uid == dr.miner.uid for dr in results)]
                if domain_model_results:
                    # Average score across all results for this model in this domain
                    scores = [r.evaluation.score for r in domain_model_results if any(r.miner.uid == dr.miner.uid for dr in results)]
                    if scores:
                        model_scores[model] = sum(scores) / len(scores)
            
            if not model_scores:
                continue
                
            # Calculate group average for this domain
            group_average = sum(model_scores.values()) / len(model_scores)
            domain_group_averages[domain] = group_average
            
            # Calculate GRPO scores: score[i] / mean(scores)
            grpo_scores = {}
            for model, score in model_scores.items():
                if group_average > 0:
                    grpo_scores[model] = score / group_average
                else:
                    grpo_scores[model] = 0.0
            
            domain_grpo_scores[domain] = grpo_scores
            
            logger.debug(f"Domain {domain}: group_avg={group_average:.4f}, models={len(grpo_scores)}")
        
        return domain_grpo_scores, domain_group_averages
    
    def calculate_skew_penalty(self, grpo_scores: Dict[str, float]) -> float:
        """
        Calculate penalty for domain skew
        
        Args:
            grpo_scores: Dict mapping domain -> GRPO score
            
        Returns:
            Penalty value (higher = more penalty)
        """
        if len(grpo_scores) < 2:
            return 0.0
        
        scores = list(grpo_scores.values())
        max_score = max(scores)
        min_score = min(scores)
        
        # Penalty is the absolute difference between domains
        penalty = abs(max_score - min_score)
        return penalty
    
    def evaluate_round(self, all_results: List[Any]) -> EvaluationRound:
        """
        Evaluate a complete round of results using GRPO
        
        Args:
            all_results: List of Result objects from all miners across all domains
            
        Returns:
            EvaluationRound with computed scores and winner
        """
        round_id = f"round_{int(time.time())}"
        round_obj = EvaluationRound(
            round_id=round_id,
            timestamp=time.time()
        )
        
        if not all_results:
            logger.warning("No results provided for evaluation")
            return round_obj
        
        # Group results by domain (using environment class name)
        domain_results = defaultdict(list)
        for result in all_results:
            domain = result.challenge.env.__class__.__name__
            domain_results[domain].append(result)
        
        # Check if we have results for all required domains
        missing_domains = set(self.required_domains) - set(domain_results.keys())
        if missing_domains:
            logger.warning(f"Missing results for required domains: {missing_domains}")
            return round_obj
        
        # Aggregate by model to collapse Sybil clones
        model_groups = self.aggregate_by_model(all_results)
        
        # Calculate GRPO scores for each domain
        domain_grpo_scores, domain_group_averages = self.calculate_grpo_scores(domain_results, model_groups)
        round_obj.domain_group_averages = domain_group_averages
        
        # Calculate final scores using min() across domains for each model
        model_final_scores = {}
        
        for model in model_groups.keys():
            # Get GRPO scores for this model across required domains
            model_grpo = {}
            for domain in self.required_domains:
                if domain in domain_grpo_scores and model in domain_grpo_scores[domain]:
                    model_grpo[domain] = domain_grpo_scores[domain][model]
                else:
                    model_grpo[domain] = 0.0
            
            # Final score is min across domains (punishes imbalance)
            final_score = min(model_grpo.values()) if model_grpo else 0.0
            
            # Apply skew penalty
            skew_penalty = self.calculate_skew_penalty(model_grpo)
            final_score_with_penalty = final_score - (self.skew_penalty_weight * skew_penalty)
            
            model_final_scores[model] = final_score_with_penalty
        
        # Determine the best performing model first
        if model_final_scores:
            winner_model_name, winner_score = max(model_final_scores.items(), key=lambda x: x[1])
            round_obj.winner_model = winner_model_name
            round_obj.winner_score = winner_score
            
            # Among all miners using the winning model, find the one with oldest block commitment
            winning_model_results = model_groups[winner_model_name]
            oldest_result = min(winning_model_results, key=lambda r: r.miner.block or float('inf'))
            round_obj.winner_uid = oldest_result.miner.uid
            
            # Create performance objects for all miners who get rewards (oldest per model)
            for model in model_groups.keys():
                model_results = model_groups[model]
                oldest_result_for_model = min(model_results, key=lambda r: r.miner.block or float('inf'))
                
                # Get GRPO scores for this model
                model_grpo = {}
                for domain in self.required_domains:
                    if domain in domain_grpo_scores and model in domain_grpo_scores[domain]:
                        model_grpo[domain] = domain_grpo_scores[domain][model]
                    else:
                        model_grpo[domain] = 0.0
                
                final_score_with_penalty = model_final_scores[model]
                
                perf = MinerPerformance(
                    uid=oldest_result_for_model.miner.uid,
                    hotkey=oldest_result_for_model.miner.hotkey,
                    model=model,
                    grpo_scores=model_grpo.copy(),
                    final_score=final_score_with_penalty
                )
                
                # Add individual scores to performance (aggregate from all miners using this model)
                for domain in self.required_domains:
                    domain_results_for_model = [r for r in model_results if r.challenge.env.__class__.__name__ == domain]
                    for result in domain_results_for_model:
                        perf.add_score(domain, result.evaluation.score)
                
                round_obj.miner_performances[oldest_result_for_model.miner.uid] = perf
            
            logger.info(f"Round {round_id}: Winner is UID {round_obj.winner_uid} (model: {round_obj.winner_model}) with score {round_obj.winner_score:.4f}")
        
        # Update cumulative miner history with EMA smoothing
        self._update_miner_history(round_obj)
        
        # Store this round
        self.evaluation_history.append(round_obj)
        
        # Save to disk
        self._save_history()
        
        return round_obj
    
    def _update_miner_history(self, round_obj: EvaluationRound):
        """Update cumulative miner history with exponential moving average"""
        for uid, current_perf in round_obj.miner_performances.items():
            if uid not in self.miner_history:
                # First time seeing this miner
                self.miner_history[uid] = MinerPerformance(
                    uid=current_perf.uid,
                    hotkey=current_perf.hotkey,
                    model=current_perf.model
                )
            
            historical_perf = self.miner_history[uid]
            
            # Update model (in case it changed)
            historical_perf.model = current_perf.model
            
            # Update GRPO scores with EMA
            for domain, current_score in current_perf.grpo_scores.items():
                if domain in historical_perf.grpo_scores:
                    # EMA: new_value = α * current + (1-α) * previous
                    historical_perf.grpo_scores[domain] = (
                        self.ema_alpha * current_score + 
                        (1 - self.ema_alpha) * historical_perf.grpo_scores[domain]
                    )
                else:
                    historical_perf.grpo_scores[domain] = current_score
            
            # Update final score
            historical_perf.update_final_score(self.required_domains)
            
            # Update domain averages and sample counts
            for domain in self.required_domains:
                if domain in current_perf.domain_averages:
                    historical_perf.domain_averages[domain] = current_perf.domain_averages[domain]
                    historical_perf.sample_counts[domain] = current_perf.sample_counts[domain]
    
    def get_leaderboard(self, limit: Optional[int] = None) -> List[MinerPerformance]:
        """
        Get current leaderboard sorted by final score
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of MinerPerformance objects sorted by final_score (descending)
        """
        leaderboard = sorted(
            self.miner_history.values(),
            key=lambda x: x.final_score,
            reverse=True
        )
        
        if limit:
            leaderboard = leaderboard[:limit]
        
        return leaderboard
    
    def get_miner_stats(self, uid: int) -> Optional[MinerPerformance]:
        """Get performance stats for a specific miner"""
        return self.miner_history.get(uid)
    
    def get_recent_rounds(self, limit: int = 10) -> List[EvaluationRound]:
        """Get recent evaluation rounds"""
        return self.evaluation_history[-limit:] if self.evaluation_history else []
    
    def clear_history(self):
        """Clear all evaluation history (useful for testing)"""
        self.evaluation_history = []
        self.miner_history = {}
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        logger.info("Cleared all evaluation history") 