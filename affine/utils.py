"""
Utility functions for the Affine framework.
Handles results processing, persistence, and display.
"""

import os
import json
import time
import logging
from collections import defaultdict
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

from .core import Result

logger = logging.getLogger("affine")
console = Console()


def print_results(results: List[Result], verbose: bool = False) -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: List of evaluation results
        verbose: Whether to show detailed output
    """
    if not results:
        console.print("❌ No results to display")
        return
    
    # Group results by miner
    stats = defaultdict(lambda: {'model': None, 'scores': [], 'errors': 0})
    for result in results:
        uid = result.miner.uid
        stats[uid]['model'] = result.miner.model
        stats[uid]['scores'].append(result.evaluation.score)
        if result.response.error:
            stats[uid]['errors'] += 1
    
    # Create results table
    table = Table(title="🏆 Evaluation Results")
    table.add_column("UID", style="cyan", justify="right")
    table.add_column("Model", style="magenta")
    table.add_column("Challenges", style="blue", justify="right")
    table.add_column("Total Score", style="green", justify="right")
    table.add_column("Average", style="yellow", justify="right")
    if verbose:
        table.add_column("Errors", style="red", justify="right")
    
    total_score = 0
    total_challenges = 0
    
    for uid, data in sorted(stats.items()):
        challenge_count = len(data['scores'])
        score_sum = sum(data['scores'])
        average = score_sum / challenge_count if challenge_count else 0
        
        total_score += score_sum
        total_challenges += challenge_count
        
        row = [
            str(uid),
            data['model'] or "Unknown",
            str(challenge_count),
            f"{score_sum:.3f}",
            f"{average:.3f}"
        ]
        
        if verbose:
            row.append(str(data['errors']))
        
        table.add_row(*row)
    
    console.print(table)
    
    # Summary statistics
    overall_average = total_score / total_challenges if total_challenges else 0
    console.print(f"\n📊 Overall Statistics:")
    console.print(f"   Total Challenges: {total_challenges}")
    console.print(f"   Total Score: {total_score:.3f}")
    console.print(f"   Average Score: {overall_average:.3f}")
    
    if verbose:
        successful = sum(1 for r in results if r.response.error is None)
        console.print(f"   Success Rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")


def save_results(results: List[Result], custom_path: str = None) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation results
        custom_path: Custom file path (optional)
        
    Returns:
        Path to saved file
    """
    if custom_path:
        file_path = custom_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    else:
        results_dir = os.path.expanduser("~/.affine/results")
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, f"{int(time.time())}.json")
    
    # Serialize results
    serialized = []
    for result in results:
        result_dict = result.model_dump()
        # Replace env object with class name for JSON serialization
        result_dict['challenge']['env'] = result.challenge.env.__class__.__name__
        serialized.append(result_dict)
    
    # Save to file
    with open(file_path, "w") as f:
        json.dump(serialized, f, indent=2, default=str)
    
    logger.info("Results saved to %s", file_path)
    console.print(f"💾 Results saved to: {file_path}")
    
    return file_path


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation results from JSON file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        List of result dictionaries
    """
    with open(file_path, "r") as f:
        results = json.load(f)
    
    logger.info("Loaded %d results from %s", len(results), file_path)
    return results


def calculate_elo_ratings(results: List[Result], k_factor: int = 32) -> Dict[str, float]:
    """
    Calculate ELO ratings from pairwise comparison results.
    
    Args:
        results: List of evaluation results
        k_factor: ELO K-factor for rating updates
        
    Returns:
        Dictionary mapping hotkeys to ELO ratings
    """
    elo = defaultdict(lambda: 1500.0)
    
    # Group results by challenge
    challenges = defaultdict(list)
    for result in results:
        challenge_id = id(result.challenge)
        challenges[challenge_id].append(result)
    
    # Process each challenge as a tournament
    for challenge_results in challenges.values():
        if len(challenge_results) < 2:
            continue
            
        # Sort by score (descending)
        sorted_results = sorted(challenge_results, key=lambda r: r.evaluation.score, reverse=True)
        
        # Update ELO for each pair
        for i, result_a in enumerate(sorted_results):
            for result_b in sorted_results[i+1:]:
                hotkey_a = result_a.miner.hotkey
                hotkey_b = result_b.miner.hotkey
                
                # Calculate expected scores
                rating_a = elo[hotkey_a]
                rating_b = elo[hotkey_b]
                expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
                expected_b = 1 - expected_a
                
                # Actual scores (1 for winner, 0 for loser, 0.5 for tie)
                if result_a.evaluation.score > result_b.evaluation.score:
                    actual_a, actual_b = 1.0, 0.0
                elif result_a.evaluation.score < result_b.evaluation.score:
                    actual_a, actual_b = 0.0, 1.0
                else:
                    actual_a, actual_b = 0.5, 0.5
                
                # Update ratings
                elo[hotkey_a] += k_factor * (actual_a - expected_a)
                elo[hotkey_b] += k_factor * (actual_b - expected_b)
    
    return dict(elo)


def print_elo_rankings(elo_ratings: Dict[str, float], top_n: int = 20) -> None:
    """
    Print ELO rankings in a formatted table.
    
    Args:
        elo_ratings: Dictionary mapping hotkeys to ELO ratings
        top_n: Number of top miners to display
    """
    # Sort by ELO rating (descending)
    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    table = Table(title=f"🏆 Top {top_n} ELO Rankings")
    table.add_column("Rank", style="bold cyan", justify="right")
    table.add_column("Hotkey", style="magenta")
    table.add_column("ELO Rating", style="green", justify="right")
    
    for rank, (hotkey, rating) in enumerate(sorted_ratings[:top_n], 1):
        table.add_row(
            str(rank),
            f"{hotkey[:8]}...{hotkey[-8:]}",
            f"{rating:.1f}"
        )
    
    console.print(table)


def summarize_challenges(results: List[Result]) -> None:
    """
    Print a summary of challenge types and performance.
    
    Args:
        results: List of evaluation results
    """
    challenge_stats = defaultdict(lambda: {'count': 0, 'total_score': 0, 'errors': 0})
    
    for result in results:
        env_name = result.challenge.env.__class__.__name__
        challenge_stats[env_name]['count'] += 1
        challenge_stats[env_name]['total_score'] += result.evaluation.score
        if result.response.error:
            challenge_stats[env_name]['errors'] += 1
    
    table = Table(title="📊 Challenge Summary")
    table.add_column("Environment", style="cyan")
    table.add_column("Count", style="blue", justify="right")
    table.add_column("Average Score", style="green", justify="right")
    table.add_column("Success Rate", style="yellow", justify="right")
    
    for env_name, stats in challenge_stats.items():
        avg_score = stats['total_score'] / stats['count']
        success_rate = (stats['count'] - stats['errors']) / stats['count'] * 100
        
        table.add_row(
            env_name,
            str(stats['count']),
            f"{avg_score:.3f}",
            f"{success_rate:.1f}%"
        )
    
    console.print(table) 