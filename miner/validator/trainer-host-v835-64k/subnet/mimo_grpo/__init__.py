"""MiMo-reference complete-group GRPO, prospective release only.

Protocol admission and authenticated checkpoint loading belong to the caller.
This package never reads miner-provided log probabilities as PPO OLD values.
"""
from .config import TrainerConfig
from .data import TaskGroup, Trajectory, InputBinding
from .training import train_update
from .projection import project_document
from .service import execute_transaction
from .recovery import restore_transaction_state

__all__ = ['TrainerConfig', 'TaskGroup', 'Trajectory', 'InputBinding', 'train_update', 'project_document', 'execute_transaction', 'restore_transaction_state']
