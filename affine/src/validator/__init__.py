"""
Validator Service - Weight Setting

Fetches scores from backend and sets weights on blockchain.
"""

from affine.backend.validator.main import ValidatorService
from affine.backend.validator.weight_setter import WeightSetter

__all__ = ["ValidatorService", "WeightSetter"]