"""
Affine - Decentralized AI Evaluation Framework
============================================

A framework for evaluating AI models through blockchain-based challenges
and decentralized mining on the Bittensor network.

Example usage:
    import affine as af
    
    # Generate challenges
    coin_env = af.COIN()
    challenges = await coin_env.many(5)
    
    # Run against miners
    results = await af.run(challenges, miners=[1, 2, 3])
    
    # Display and save results
    af.print_results(results)
    af.save_results(results)
    
    # Or use the CLI
    af.cli()
"""

# Import core functionality from modular components
from .core import (
    Miner,
    Response, 
    BaseEnv,
    Challenge,
    Evaluation,
    Result,
    get_chute,
    miners,
    run,
    get_conf,  # Legacy function
)

# Import configuration and logging
from .config import config, setup_logging

# Import utilities
from .utils import (
    print_results,
    save_results,
    load_results,
    calculate_elo_ratings,
    print_elo_rankings,
    summarize_challenges,
)

# Import CLI
from .cli import cli

# Import environments for easy access
from .envs.COIN import COIN
from .envs.SAT import SAT
from .envs.SAT1 import SAT1

# Version and metadata
__version__ = "0.2.0"
__author__ = "Affine Foundation"
__description__ = "Decentralized AI Evaluation Framework"

# Make environments easily accessible
ENVS = {
    "COIN": COIN,
    "SAT": SAT,
    "SAT1": SAT1,
}

# Expose main interfaces
__all__ = [
    # Core classes and functions
    'Miner', 'Response', 'BaseEnv', 'Challenge', 'Evaluation', 'Result',
    'get_chute', 'miners', 'run', 'get_conf',
    
    # Configuration
    'config', 'setup_logging',
    
    # Utilities
    'print_results', 'save_results', 'load_results',
    'calculate_elo_ratings', 'print_elo_rankings', 'summarize_challenges',
    
    # Environments and CLI
    'COIN', 'SAT', 'SAT1', 'ENVS', 'cli'
]