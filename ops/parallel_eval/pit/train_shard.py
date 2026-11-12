"""Host-side view of the sharded (ZeRO-3) trainer for GPUs below 100 GiB (8x RTX5090, 32 GiB each).

The code that runs on the box is ``host_v834f_signing/stage/v835_zero3.py``, a HOST_FILE of host-v834f. It is not
part of ``miner/validator/trainer-v834e`` (that tree stays byte-identical to the ROOT-admitted host-v834e).

This module gives the PIT host: the threshold predicate, the one environment switch the box trainer reads
(``PIT_ZERO3``), the box module's memory model, and two checks that run the real stage code in real processes on
the CPU (torch FSDP over gloo). ``PIT_FP32_ACCUM_DEVICE`` and ``PIT_LOW_MEM_SHARD`` no longer exist: nothing reads
them, so they cannot change any behaviour.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Mapping, Optional

from . import LOW_MEM_GPU_GIB

PIT_ZERO3_ENV = 'PIT_ZERO3'
PIT_ZERO3_PROFILE_ENV = 'PIT_ZERO3_PROFILE'      # names one set of sharded-update settings (v835_zero3.PROFILES)
SIGNING = Path(__file__).resolve().parent / 'host_v834f_signing'
STAGE_V835 = SIGNING / 'stage' / 'v835_zero3.py'
NUMERICS = SIGNING / 'harness' / 'zero3_numerics.py'


def _load_v835():
    """The box module, read from the stage for its policy constants and memory model (needs torch).

    Compiled from source here: the import system would leave a bytecode file in stage/, and the signer refuses a
    stage that holds any file the reviewed manifest does not list.
    """
    if not STAGE_V835.is_file():
        raise RuntimeError('v835_zero3.py missing from host_v834f_signing/stage')
    module = types.ModuleType('v835_zero3_host')
    module.__file__ = str(STAGE_V835)
    exec(compile(STAGE_V835.read_bytes(), str(STAGE_V835), 'exec'), module.__dict__)
    return module


def needs_optimizer_shard(per_gpu_mem_gib: Optional[float]) -> bool:
    """True when one GPU cannot hold the unsharded update (bf16 actor + bf16 gradient + fp32 accumulator)."""
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
    return _load_v835().per_gpu_mem_gib()


def configure_low_mem_train(per_gpu_mem_gib: Optional[float] = None) -> Mapping[str, Any]:
    """Set or clear ``PIT_ZERO3`` for the box trainer of this process tree; return the chosen policy."""
    mem = per_gpu_mem_gib if per_gpu_mem_gib is not None else detect_per_gpu_mem_gib()
    shard = needs_optimizer_shard(mem)
    if shard:
        os.environ[PIT_ZERO3_ENV] = '1'
    else:
        os.environ.pop(PIT_ZERO3_ENV, None)
    return {'per_gpu_mem_gib': mem, 'zero3': shard, 'threshold_gib': float(LOW_MEM_GPU_GIB)}


def estimate_peak_gib(**kwargs) -> Mapping[str, Any]:
    """Per-GPU peak of one sharded micro-batch (see v835_zero3.estimate_peak_gib). ``n_params`` scales the 9B
    geometry when no loaded model inventory is given."""
    module = _load_v835()
    inv = kwargs.pop('inv', None) or module.scaled_inventory(kwargs.pop('n_params', 9.4e9))
    kwargs.pop('n_params', None)
    return module.estimate_peak_gib(inv, **kwargs)


def preflight_or_raise(**kwargs) -> Mapping[str, Any]:
    return _load_v835().preflight_or_raise(**kwargs)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def run_numerics(*, world: int, dtype: str, seed: int = 0, timeout: float = 900.0, offload_inputs: bool = False,
                 work: Optional[Path] = None, profile: Optional[str] = None,
                 host_block_bytes: Optional[int] = None) -> Mapping[str, Any]:
    """Run ``harness/zero3_numerics.py``: ``world`` processes shard a tiny production-class model with the stage's
    own code and one gradient is compared with the sum that the settings of the run define. Returns the tool's
    JSON result.

    ``offload_inputs``: every rank keeps its checkpointed layer inputs behind the stage's host-memory hooks.
    ``work``: keep the tool's files there (the gathered gradient is ``result['gradient_file']``); by default they
    go to a scratch directory that is removed.
    ``profile``: the name of the settings (``v835_zero3.PROFILES``); None = the stage's default.
    ``host_block_bytes``: size of one host block of the layer inputs; None = the stage's own size, which no
    tensor of the tiny model reaches.
    """
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in (PIT_ZERO3_ENV, PIT_ZERO3_PROFILE_ENV):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix='zero3-numerics-') as scratch:
        argv = [sys.executable, '-B', str(NUMERICS), '--world', str(world), '--dtype', dtype, '--seed', str(seed),
                '--port', str(_free_port()), '--work', str(work or scratch)]
        if offload_inputs:
            argv.append('--offload-inputs')
        if profile:
            argv += ['--profile', profile]
        if host_block_bytes:
            argv += ['--host-block-bytes', str(host_block_bytes)]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    lines = [line for line in proc.stdout.splitlines() if line.startswith('{')]
    if not lines:
        raise RuntimeError('zero3_numerics produced no result (rc %s): %s' % (proc.returncode, proc.stderr[-2000:]))
    result = json.loads(lines[-1])
    result['returncode'] = proc.returncode
    return result


def numeric_equivalence_step(*, seed: int = 0, profile: Optional[str] = None) -> Mapping[str, Any]:
    """One gradient in the production precision (bf16 forward) at world 2: the fp32 gradient the sharded path
    hands to the optimizer against the sum that the settings define, built from the same autograd gradients."""
    result = dict(run_numerics(world=2, dtype='bfloat16', seed=seed, profile=profile))
    if not result.get('ok'):
        raise RuntimeError('sharded bf16 gradient check failed: %s' % json.dumps(result)[:1500])
    reduction = result['reduction']
    result.update(max_abs_diff=reduction['max_abs_diff'],
                  norm_diff=reduction['rel_diff'] * reduction['gradient_norm'])
    return result


def numeric_equivalence_zero3(*, seed: int = 0, world: int = 8) -> Mapping[str, Any]:
    """The sharded gradient of ``world`` ranks against one unsharded process, in fp32 (the precision in which the
    CPU kernels are reproducible between processes; see harness/zero3_numerics.py)."""
    return run_numerics(world=world, dtype='float32', seed=seed)
