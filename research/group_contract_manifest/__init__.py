"""Prospective manifest dispatch; not connected to production."""
from .resolver import (
    COMPLETE_GROUP_SAMPLING_VERSION,
    GROUP_VERSION,
    REQUIRED_CLIENT_FEATURES,
    CompleteGroupContract,
    LegacyQuotaContract,
    ValidationHooks,
    for_manifest,
)

__all__ = [
    'COMPLETE_GROUP_SAMPLING_VERSION', 'GROUP_VERSION', 'REQUIRED_CLIENT_FEATURES',
    'CompleteGroupContract', 'LegacyQuotaContract', 'ValidationHooks', 'for_manifest',
]
