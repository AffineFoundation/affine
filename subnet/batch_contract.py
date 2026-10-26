"""Prospective composed-source adapter for authenticated batch manifests.

Importing or resolving a complete-group contract performs no model, network,
filesystem or signing operation. Historical validators are imported lazily from
this same authenticated source bundle, never selected from miner input.
Signature, source and calibration-evidence admission remain caller obligations.
"""
from .complete_group_contract import (
    COMPLETE_GROUP_SAMPLING_VERSION,
    GROUP_VERSION,
    REQUIRED_CLIENT_FEATURES,
    CompleteGroupContract,
    LegacyQuotaContract,
    ValidationHooks,
    for_manifest as _resolve,
)


def _configured_quotas(config):
    # Canonical historical client has no batch_quotas module. Do not silently
    # project a new samples_per_batch setting into its original K/L protocol.
    if 'samples_per_batch' in config:
        count = config['samples_per_batch']
        if type(count) is not int or not 4 <= count <= 128 or count % 2:
            raise ValueError('bounded even historical sample count')
        if any(k in config and (type(config[k]) is not int or config[k] != count // 2)
               for k in ('K', 'L')):
            raise ValueError('historical quota conflict')
        return count // 2, count // 2
    return config.get('K', 1), config.get('L', 1)


def _class_quotas(k, l, sampling):
    if any(type(v) is not int or not 1 <= v <= 64 for v in (k, l)):
        raise ValueError('bounded historical class quotas')
    return k, l


def _validate_sampling(sampling):
    from .forced_sampling import validate
    return validate(sampling)


def _validate_calibration(calibration):
    from .fast_prefill_audit import calibration as validate
    return validate(calibration)


def for_manifest(manifest, *, supported_features=()):
    """Resolve only after manifest/source authentication, before model loading.

    Feature support is explicit at the call site. Merely importing this adapter
    must not cause an old pair-based consumer to claim complete-group support.
    No projected K/L aliases are provided for complete groups.
    """
    hooks = ValidationHooks(_configured_quotas, _class_quotas,
                            _validate_sampling, _validate_calibration)
    return _resolve(manifest, validators=hooks, supported_features=supported_features)


__all__ = [
    'COMPLETE_GROUP_SAMPLING_VERSION', 'GROUP_VERSION', 'REQUIRED_CLIENT_FEATURES',
    'CompleteGroupContract', 'LegacyQuotaContract', 'for_manifest',
]
