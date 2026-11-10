"""Box-side train entry (BRIEF §1.4).

This module is the portable surface that runs ON the box. It does not import
the heavy trainer at module import time. Hard asserts:

* ``AFFINE_FP32_EXPECTED_MASTER_HASH`` present and hex64
* ``AFFINE_FP32_BOOTSTRAP_OVERRIDE`` unset
* no ``FP32_BOOTSTRAP_OVERRIDE`` module override

The actual v834e train call is injected for unit tests; on a real box the
operator wires ``miner.validator.trainer-v834e`` via the signed release.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping, Optional


class TrainBoxError(RuntimeError):
    pass


def assert_no_bootstrap_override(env: Optional[Mapping[str, str]] = None) -> None:
    env = env or os.environ
    if env.get('AFFINE_FP32_BOOTSTRAP_OVERRIDE'):
        raise TrainBoxError('AFFINE_FP32_BOOTSTRAP_OVERRIDE must be unset on PIT boxes')
    if env.get('FP32_BOOTSTRAP_OVERRIDE'):
        raise TrainBoxError('FP32_BOOTSTRAP_OVERRIDE must be unset on PIT boxes')


def assert_expected_master_hash(env: Optional[Mapping[str, str]] = None) -> str:
    env = env or os.environ
    dig = env.get('AFFINE_FP32_EXPECTED_MASTER_HASH') or env.get('FP32_EXPECTED_MASTER')
    if not isinstance(dig, str) or len(dig) != 64 or any(c not in '0123456789abcdef' for c in dig):
        raise TrainBoxError('AFFINE_FP32_EXPECTED_MASTER_HASH must be lowercase hex64')
    return dig


def run_train_box(
    *,
    train_fn: Callable[..., Mapping[str, Any]],
    env: Optional[Mapping[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run one train attempt under PIT hard asserts."""
    assert_no_bootstrap_override(env)
    expected = assert_expected_master_hash(env)
    if kwargs.get('fp32_bootstrap_override') or kwargs.get('bootstrap_override'):
        raise TrainBoxError('bootstrap override argument refused')
    out = dict(train_fn(expected_master_digest=expected, **kwargs))
    if out.get('used_fp32_bootstrap_override'):
        raise TrainBoxError('trainer reported bootstrap override')
    src = out.get('fp32_master_source') or out.get('fp32_master', {}).get('source')
    if src is not None and src not in ('disk', 'memory'):
        raise TrainBoxError('fp32_master.source must be disk|memory')
    return out
