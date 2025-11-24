"""
Validator Service - Weight Setting

Fetches scores from backend and sets weights on blockchain.
"""

from affine.src.validator.main import ValidatorService
from affine.src.validator.weight_setter import WeightSetter

__all__ = ["ValidatorService", "WeightSetter"]