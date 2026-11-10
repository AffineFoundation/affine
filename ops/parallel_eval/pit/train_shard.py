"""Low-memory train path for <100 GiB GPUs (RTX5090 32 GiB).

CPUAdamW already keeps the FP32 master + Adam state on CPU. The remaining
GPU-side cost is the full FP32 gradient accumulator (~4 bytes × params). On a
9B model that is ~36 GiB and will not fit next to the BF16 actor on 32 GiB.

When per-GPU HBM is under ``LOW_MEM_GPU_GIB`` (100), enable ZeRO-style offload:
move the FP32 accumulator to CPU. Math is identical (same add / clip / scale);
only the buffer device changes. Optional parameter-group sharding across DP
ranks is available for true ZeRO-2 when ``world_size > 1``.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import LOW_MEM_GPU_GIB

# Env the on-box train_pit / host trainer reads.
PIT_LOW_MEM_ENV = 'PIT_LOW_MEM_SHARD'
PIT_ACCUM_DEVICE_ENV = 'PIT_FP32_ACCUM_DEVICE'


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
    """Best-effort local HBM size in GiB (nvidia-smi); None when unavailable."""
    env = per_gpu_mem_gib_from_env()
    if env is not None:
        return env
    try:
        import subprocess
        out = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.total',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0:
            return None
        line = (out.stdout or '').strip().splitlines()[0]
        # nvidia-smi reports MiB.
        return float(line) / 1024.0
    except Exception:
        return None


def configure_low_mem_train(per_gpu_mem_gib: Optional[float] = None) -> Mapping[str, Any]:
    """Set process env for the on-box trainer; return the chosen policy."""
    mem = per_gpu_mem_gib if per_gpu_mem_gib is not None else detect_per_gpu_mem_gib()
    shard = needs_optimizer_shard(mem)
    if shard:
        os.environ[PIT_LOW_MEM_ENV] = '1'
        os.environ[PIT_ACCUM_DEVICE_ENV] = 'cpu'
    else:
        os.environ.pop(PIT_LOW_MEM_ENV, None)
        os.environ.pop(PIT_ACCUM_DEVICE_ENV, None)
    return {
        'per_gpu_mem_gib': mem,
        'low_mem_shard': shard,
        'fp32_accum_device': 'cpu' if shard else None,
        'threshold_gib': float(LOW_MEM_GPU_GIB),
    }


def fp32_accum_device_from_env() -> Optional[str]:
    if os.environ.get(PIT_LOW_MEM_ENV) in ('1', 'true', 'True', 'yes'):
        return os.environ.get(PIT_ACCUM_DEVICE_ENV) or 'cpu'
    return os.environ.get(PIT_ACCUM_DEVICE_ENV) or None


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
        # Stable ownership independent of process ordering.
        h = sum(ord(c) for c in name) % int(world_size)
        if h == int(rank):
            owned.append(name)
    # Guarantee every rank owns at least one when possible.
    if not owned and names:
        owned.append(names[int(rank) % len(names)])
    return owned


def numeric_equivalence_step(
    *,
    shapes: Sequence[Tuple[int, ...]] = ((8, 8), (4,)),
    clip: float = 1.0,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Compare default GPU-buffer path vs CPU-offload path on a tiny model.

    Returns max abs diff of clipped buffers; must be ~0 (same math).
    """
    import torch
    from subnet.mimo_grpo.optimizer import FP32Accumulator

    torch.manual_seed(seed)
    params_a = []
    params_b = []
    for i, shape in enumerate(shapes):
        p = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16))
        params_a.append(('p%d' % i, p))
        params_b.append(('p%d' % i, p.clone().detach().requires_grad_(True)))

    # Plant identical grads.
    for (_, a), (_, b) in zip(params_a, params_b):
        g = torch.randn_like(a, dtype=torch.float32)
        a.grad = g.to(dtype=a.dtype)
        b.grad = g.to(dtype=b.dtype).clone()

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
