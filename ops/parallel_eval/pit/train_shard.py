"""Low-memory / ZeRO-3 train helpers for <100 GiB GPUs (RTX5090 32 GiB).

The on-box path that actually runs is ``host-v834f/v835_zero3.py`` (staged under
``ops/parallel_eval/pit/host_v834f_signing/stage/``). This module mirrors the
public API for host-side tests and config. ``v835_zero3.py`` must NOT live in
``trainer-v834e/`` (that tree stays byte-identical to host_v834e_admitted).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from . import LOW_MEM_GPU_GIB

PIT_LOW_MEM_ENV = 'PIT_LOW_MEM_SHARD'
PIT_ACCUM_DEVICE_ENV = 'PIT_FP32_ACCUM_DEVICE'
PIT_ZERO3_ENV = 'PIT_ZERO3'


def _stage_v835_path() -> Path:
    return Path(__file__).resolve().parent / 'host_v834f_signing' / 'stage' / 'v835_zero3.py'


def _load_v835():
    path = _stage_v835_path()
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location('v835_zero3_host', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def needs_optimizer_shard(per_gpu_mem_gib: Optional[float]) -> bool:
    """True when a full per-GPU FP32 master/accumulator will not fit."""
    if per_gpu_mem_gib is None:
        return False
    try:
        return float(per_gpu_mem_gib) < float(LOW_MEM_GPU_GIB)
    except (TypeError, ValueError):
        return False


def per_gpu_mem_gib_from_env() -> Optional[float]:
    for key in ('PIT_PER_GPU_MEM_GIB', 'CUDA_VISIBLE_MEM_GIB'):
        raw = os.environ.get(key)
        if raw:
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def detect_per_gpu_mem_gib() -> Optional[float]:
    env = per_gpu_mem_gib_from_env()
    if env is not None:
        return env
    z = _load_v835()
    if z is not None:
        return z.per_gpu_mem_gib()
    return None


def configure_low_mem_train(per_gpu_mem_gib: Optional[float] = None) -> Mapping[str, Any]:
    """Set process env for the on-box trainer; return the chosen policy."""
    mem = per_gpu_mem_gib if per_gpu_mem_gib is not None else detect_per_gpu_mem_gib()
    shard = needs_optimizer_shard(mem)
    if shard:
        os.environ[PIT_LOW_MEM_ENV] = '1'
        os.environ[PIT_ACCUM_DEVICE_ENV] = 'cpu'
        os.environ[PIT_ZERO3_ENV] = '1'
    else:
        os.environ.pop(PIT_LOW_MEM_ENV, None)
        os.environ.pop(PIT_ACCUM_DEVICE_ENV, None)
        os.environ.pop(PIT_ZERO3_ENV, None)
    return {
        'per_gpu_mem_gib': mem,
        'low_mem_shard': shard,
        'zero3': shard,
        'fp32_accum_device': 'cpu' if shard else None,
        'threshold_gib': float(LOW_MEM_GPU_GIB),
    }


def fp32_accum_device_from_env() -> Optional[str]:
    """CPU accum device only when PIT_ZERO3 is set (never from ACCUM env alone)."""
    if os.environ.get(PIT_ZERO3_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return os.environ.get(PIT_ACCUM_DEVICE_ENV) or 'cpu'
    return None


def shard_param_names(
    names: Sequence[str],
    *,
    rank: int,
    world_size: int,
) -> List[str]:
    """ZeRO-2 style ownership: rank ``i`` owns names where hash % world == i."""
    if world_size < 1:
        raise ValueError('world_size must be >= 1')
    if not (0 <= int(rank) < int(world_size)):
        raise ValueError('rank out of range')
    if world_size == 1:
        return list(names)
    owned = []
    for name in names:
        h = sum(ord(c) for c in name) % int(world_size)
        if h == int(rank):
            owned.append(name)
    if not owned and names:
        owned.append(names[int(rank) % len(names)])
    return owned


def estimate_peak_gib(**kwargs) -> Mapping[str, float]:
    z = _load_v835()
    if z is None:
        raise RuntimeError('v835_zero3.py missing from host_v834f_signing/stage')
    return z.estimate_peak_gib(**kwargs)


def preflight_or_raise(**kwargs) -> Mapping[str, Any]:
    z = _load_v835()
    if z is None:
        raise RuntimeError('v835_zero3.py missing from host_v834f_signing/stage')
    return z.preflight_or_raise(**kwargs)


def numeric_equivalence_step(
    *,
    shapes: Sequence[Tuple[int, ...]] = ((8, 8), (4,)),
    clip: float = 1.0,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Compare default GPU-buffer path vs CPU-offload path on a tiny model.

    Applies the v835 accumulator patch under a temporary PIT_ZERO3 so the
    golden-bundle FP32Accumulator gains ``buffer_device`` for the comparison.
    """
    import torch
    from subnet.mimo_grpo.optimizer import FP32Accumulator

    z = _load_v835()
    prev = os.environ.get(PIT_ZERO3_ENV)
    os.environ[PIT_ZERO3_ENV] = '1'
    try:
        if z is not None:
            z.patch_fp32_accumulator()
        torch.manual_seed(seed)
        params_a = []
        params_b = []
        for i, shape in enumerate(shapes):
            p = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16))
            params_a.append(('p%d' % i, p))
            params_b.append(('p%d' % i, p.clone().detach().requires_grad_(True)))

        for (_, a), (_, b) in zip(params_a, params_b):
            g = torch.randn_like(a, dtype=torch.float32)
            a.grad = g.to(dtype=a.dtype)
            b.grad = g.to(dtype=b.dtype).clone()

        # Default path: no buffer_device (and with patch + ZERO3 the default
        # becomes cpu — compare explicit None via orig when available).
        Acc = FP32Accumulator
        if getattr(Acc, '_v835_orig_init', None) is not None:
            acc_a = object.__new__(Acc)
            Acc._v835_orig_init(acc_a, params_a)
        else:
            # Repo subnet already accepts buffer_device=None.
            acc_a = FP32Accumulator(params_a)
        acc_b = FP32Accumulator(params_b, buffer_device='cpu')
        acc_a.capture()
        acc_b.capture()
        n_a = acc_a.clip(clip)
        n_b = acc_b.clip(clip)
        diffs = []
        for name, _ in params_a:
            diffs.append(float((acc_a.buffers[name].cpu() - acc_b.buffers[name].cpu()).abs().max()))
        return {
            'max_abs_diff': max(diffs) if diffs else 0.0,
            'norm_a': n_a,
            'norm_b': n_b,
            'norm_diff': abs(n_a - n_b),
        }
    finally:
        if prev is None:
            os.environ.pop(PIT_ZERO3_ENV, None)
        else:
            os.environ[PIT_ZERO3_ENV] = prev


def numeric_equivalence_zero3(**kwargs) -> Mapping[str, Any]:
    """Tiny-model single-GPU vs sharded ZeRO-3-style update (final bf16 must match)."""
    z = _load_v835()
    if z is None:
        raise RuntimeError('v835_zero3.py missing from host_v834f_signing/stage')
    return z.numeric_equivalence_tiny(**kwargs)
