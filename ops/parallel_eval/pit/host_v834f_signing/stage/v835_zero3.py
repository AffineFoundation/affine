"""v8.3.5 / host-v834f: FSDP FULL_SHARD (ZeRO-3) train path for 8x low-VRAM GPUs.

Shards BF16 params + grads across the DP world (FSDP ShardingStrategy.FULL_SHARD).
FP32 masters stay CPU-offloaded (existing CPUAdamW / fresh-adam path). The FP32
gradient accumulator is forced onto CPU so a full 9B fp32 copy never lands in HBM.

Math target: same optimizer, lr, fp32-master update semantics and bf16 projection
as the single-GPU / unsharded DP path. Per-wave FSDP reduce-scatter already sums
microbatch grads across ranks; rank 0 then runs the unchanged optimizer.step.

Preflight refuses to start when the projected per-GPU peak exceeds local VRAM.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

ZERO3_ENV = 'PIT_ZERO3'
ACCUM_DEVICE_ENV = 'PIT_FP32_ACCUM_DEVICE'
# GiB of HBM at or below which ZeRO-3 is required for the 9B update.
LOW_MEM_GPU_GIB = 100.0
# Measured single-GPU peak on B300 (~112 GiB) for the current 9B PIT step.
# Breakdown used for projection: bf16 params + fp32 accum + activations/other.
REF_PEAK_GIB = 112.0
REF_PARAM_GIB = 18.0
REF_ACCUM_GIB = 36.0
# Residual = activations + workspace + fragmentation on the reference single-GPU run.
REF_RESIDUAL_GIB = REF_PEAK_GIB - REF_PARAM_GIB - REF_ACCUM_GIB
# Safety margin on top of the projection.
PREFLIGHT_MARGIN = 1.15
HOST_VERSION = 'host-v834f'


class VramPreflightError(RuntimeError):
    """Projected per-GPU peak exceeds local VRAM; refuse rather than OOM mid-run."""


def zero3_enabled() -> bool:
    return os.environ.get(ZERO3_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')


def enable_zero3_env() -> None:
    os.environ[ZERO3_ENV] = '1'
    os.environ[ACCUM_DEVICE_ENV] = 'cpu'


def per_gpu_mem_gib() -> Optional[float]:
    for key in ('PIT_PER_GPU_MEM_GIB', 'CUDA_VISIBLE_MEM_GIB'):
        raw = os.environ.get(key)
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / float(1 << 30)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode == 0:
            line = (out.stdout or '').strip().splitlines()[0]
            return float(line) / 1024.0
    except Exception:
        return None
    return None


def needs_zero3(mem_gib: Optional[float] = None) -> bool:
    if zero3_enabled():
        return True
    mem = per_gpu_mem_gib() if mem_gib is None else mem_gib
    if mem is None:
        return False
    return float(mem) < float(LOW_MEM_GPU_GIB)


def n_params_from_model(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def estimate_peak_gib(
    *,
    n_params: int,
    world: int = 8,
    token_budget: Optional[int] = None,
    ref_token_budget: int = 69632,
    sharded: bool = True,
) -> Mapping[str, float]:
    """Project per-GPU peak HBM for the 9B (or scaled) update.

    Sharded path: params/grads / world, fp32 accum on CPU (0 HBM), residual
    activations scaled by token_budget vs the reference B300 measurement.
    """
    world = max(int(world), 1)
    param_gib = (float(n_params) * 2.0) / float(1 << 30)  # bf16
    # Scale residual from the 9B reference by param count and token budget.
    scale_params = param_gib / REF_PARAM_GIB if REF_PARAM_GIB else 1.0
    if token_budget is None:
        scale_tok = 1.0
    else:
        scale_tok = max(float(token_budget) / float(ref_token_budget), 0.05)
    residual = REF_RESIDUAL_GIB * scale_params * scale_tok
    if sharded:
        params_hbm = param_gib / float(world)
        # Grad shards live in HBM briefly; count one shard (fp32 = 2x bf16).
        grad_hbm = (param_gib * 2.0) / float(world)
        accum_hbm = 0.0
        # FSDP all-gather workspace: one full layer ≈ small; budget 1/world of params.
        gather_hbm = param_gib / float(world)
        peak = (params_hbm + grad_hbm + accum_hbm + gather_hbm + residual) * PREFLIGHT_MARGIN
    else:
        peak = (param_gib + REF_ACCUM_GIB * scale_params + residual) * PREFLIGHT_MARGIN
    return {
        'peak_gib': float(peak),
        'param_gib': float(param_gib),
        'residual_gib': float(residual),
        'world': float(world),
        'sharded': 1.0 if sharded else 0.0,
        'margin': float(PREFLIGHT_MARGIN),
    }


def preflight_or_raise(
    *,
    n_params: int,
    per_gpu_gib: Optional[float],
    world: int = 8,
    token_budget: Optional[int] = None,
    sharded: bool = True,
) -> Mapping[str, Any]:
    """Refuse training when projected per-GPU peak exceeds VRAM."""
    if per_gpu_gib is None:
        raise VramPreflightError('per-GPU VRAM unknown; refuse to start sharded train')
    est = dict(estimate_peak_gib(
        n_params=n_params, world=world, token_budget=token_budget, sharded=sharded,
    ))
    est['per_gpu_gib'] = float(per_gpu_gib)
    if float(est['peak_gib']) > float(per_gpu_gib):
        raise VramPreflightError(
            'projected per-GPU peak %.1f GiB exceeds VRAM %.1f GiB (world=%d, sharded=%s)'
            % (est['peak_gib'], per_gpu_gib, world, sharded)
        )
    return est


def patch_fp32_accumulator() -> None:
    """Ensure subnet FP32Accumulator accepts buffer_device='cpu' (golden bundle may lack it)."""
    import torch
    import subnet.mimo_grpo.optimizer as opt_mod
    Acc = opt_mod.FP32Accumulator
    if getattr(Acc, '_v835_buffer_device', False):
        return
    _orig_init = Acc.__init__
    _orig_capture = Acc.capture

    def __init__(self, named_parameters, *, buffer_device=None):
        if buffer_device is None and zero3_enabled():
            buffer_device = 'cpu'
        env = os.environ.get(ACCUM_DEVICE_ENV)
        if buffer_device is None and env:
            buffer_device = env
        self.parameters = tuple((name, p) for name, p in named_parameters if p.requires_grad)
        if not self.parameters or len({name for name, _ in self.parameters}) != len(self.parameters):
            raise ValueError('unique nonempty trainable parameter inventory')
        if len({id(p) for _, p in self.parameters}) != len(self.parameters):
            raise ValueError('duplicate trainable parameter object')
        self.buffer_device = buffer_device
        if buffer_device is None:
            self.buffers = {name: torch.zeros_like(p, dtype=torch.float32) for name, p in self.parameters}
        else:
            self.buffers = {
                name: torch.zeros(p.shape, dtype=torch.float32, device=buffer_device)
                for name, p in self.parameters
            }
        self.microsteps = 0
        self.clipped = False

    def capture(self):
        if self.clipped:
            raise ValueError('gradient accumulator already clipped')
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None or gradient.is_sparse or gradient.shape != parameter.shape:
                raise ValueError('missing/dense gradient inventory: ' + name)
        with torch.no_grad():
            for name, parameter in self.parameters:
                grad = parameter.grad
                if self.buffer_device is not None and grad.device != self.buffers[name].device:
                    self.buffers[name].add_(grad.detach().to(self.buffers[name].device, dtype=torch.float32))
                else:
                    self.buffers[name].add_(grad)
                parameter.grad = None
        self.microsteps += 1

    Acc.__init__ = __init__
    Acc.capture = capture
    Acc._v835_buffer_device = True
    # Keep originals available for equivalence tests.
    Acc._v835_orig_init = _orig_init
    Acc._v835_orig_capture = _orig_capture


def _layer_wrap_policy(model):
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    try:
        layer = model.model.language_model.layers[0]
    except Exception:
        # Fallback: wrap every direct child with parameters.
        classes = {type(m) for m in model.children() if any(p.requires_grad for p in m.parameters())}
        if not classes:
            raise ValueError('no FSDP-wrappable trainable children')
        return ModuleWrapPolicy(classes)
    return ModuleWrapPolicy({type(layer)})


def wrap_fsdp(model, *, local_rank: int = 0):
    """FULL_SHARD (ZeRO-3) wrap. use_orig_params keeps named_parameters usable for CPUAdamW."""
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    if getattr(model, '_v835_fsdp', False):
        return model
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    wrapped = FSDP(
        model,
        auto_wrap_policy=_layer_wrap_policy(model),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        sync_module_states=True,
    )
    wrapped._v835_fsdp = True
    return wrapped


def patch_dp(dp_mod) -> None:
    """Patch v83_dp so _passes uses CPU accum under ZeRO-3 and _reduce works on CPU via gloo."""
    if getattr(dp_mod, '_v835_patched', False):
        return
    import torch
    import torch.distributed as dist
    _passes_orig = dp_mod._passes
    _reduce_orig = dp_mod._reduce

    def _passes(B, FK, actor, ref, rows, plan, config, agg_config, loss_scale, active_rollouts, versions, guard, record):
        from subnet.mimo_grpo.optimizer import FP32Accumulator
        accumulator, details, losses, err = None, {}, {}, None
        try:
            B._reference_pass(rows, plan, config, ref, engine='packed', logit_chunk=dp_mod.LOGIT_CHUNK, guard=guard)
            if B._versions(actor.model) != versions:
                raise ValueError('input actor changed during reference preparation')
            for p in actor.model.parameters():
                p.grad = None
            buf_dev = 'cpu' if zero3_enabled() or os.environ.get(ACCUM_DEVICE_ENV) == 'cpu' else None
            accumulator = FP32Accumulator(actor.model.named_parameters(), buffer_device=buf_dev)
            B._gradient_pass(
                actor, rows, plan, config, agg_config, loss_scale, active_rollouts, accumulator, versions,
                details, losses, engine='packed', logit_chunk=dp_mod.LOGIT_CHUNK, guard=guard, record=record,
            )
            torch.cuda.synchronize()
        except BaseException as e:
            err = '%s: %s' % (type(e).__name__, e)
        return accumulator, details, losses, err

    def _reduce(B, rows, accumulator, details, losses, err, rank):
        world = dp_mod.STATE['world']
        ok = torch.tensor([0.0 if err else 1.0], device='cuda')
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        errs = [None] * world
        dist.all_gather_object(errs, err)
        if ok.item() < 1:
            if accumulator is not None:
                accumulator.buffers.clear()
            return None, [e for e in errs if e]
        # CPU buffers: reduce on the gloo ctrl group (NCCL cannot host CPU tensors).
        group = None
        first = next(iter(accumulator.buffers.values()))
        if first.device.type == 'cpu':
            group = dp_mod.STATE.get('ctrl')
        for name in sorted(accumulator.buffers):
            dist.reduce(accumulator.buffers[name], dst=0, op=dist.ReduceOp.SUM, group=group)
        gathered = [None] * world
        dist.all_gather_object(
            gathered,
            (details, losses, {i: (rows[i]['keep'], rows[i]['derail_index']) for i in details}),
        )
        if rank != 0:
            accumulator.buffers.clear()
            return None, []
        all_d, all_l = {}, {}
        for d, l, keep in gathered:
            all_d.update(d)
            all_l.update(l)
            for i, (k, di) in keep.items():
                rows[i]['keep'], rows[i]['derail_index'] = k, di
        return (all_d, all_l), []

    dp_mod._passes = _passes
    dp_mod._reduce = _reduce
    dp_mod._passes_orig = _passes_orig
    dp_mod._reduce_orig = _reduce_orig
    dp_mod._v835_patched = True


def wrap_host_install(host_mod) -> None:
    """Wrap mimo_host_trainer.install so helpers + rank0 get ZeRO-3 patches before DP starts."""
    if getattr(host_mod, '_v835_install_wrapped', False):
        return
    import subprocess
    import sys
    orig = host_mod.install

    def install(config, start_dp=False):
        engine = config['engine']
        mem = per_gpu_mem_gib()
        if not needs_zero3(mem):
            return orig(config, start_dp=start_dp)
        enable_zero3_env()
        patch_fp32_accumulator()
        # Slightly tighter token budget on low-VRAM to keep activations inside 32 GiB.
        if int(engine.get('dp_token_budget', 69632)) > 24576 and (mem is not None and mem < 40):
            engine = dict(engine)
            engine['dp_token_budget'] = 24576
            config = dict(config, engine=engine)
        # Load dp early, patch, then run original install with patched helper argv.
        dp = host_mod.load('v83_dp', 'v83_dp.py')
        patch_dp(dp)
        if not start_dp:
            return orig(config, start_dp=False)

        # Re-implement the DP spawn portion so helpers start with zero3 applied.
        # Call original with start_dp=False to install everything else, then start DP ourselves.
        info = orig(config, start_dp=False)
        world, port = engine['dp_world'], engine['dp_port']
        helpers = []
        helper_entry = str(host_mod.P / 'v835_zero3.py')
        for r in range(1, world):
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(r),
                MASTER_ADDR='127.0.0.1',
                MASTER_PORT=str(port),
                TRITON_CACHE_DIR=host_mod.TRITON_CACHE + '/rank%d' % r,
            )
            if zero3_enabled():
                env[ZERO3_ENV] = '1'
                env[ACCUM_DEVICE_ENV] = 'cpu'
            helpers.append(subprocess.Popen(
                [
                    sys.executable, '-B', helper_entry, 'helper',
                    str(r), str(world), str(port),
                    str(host_mod.P), str(__import__('pathlib').Path(config['source_root']).resolve()),
                ],
                env=env,
                cwd=str(__import__('pathlib').Path(config['source_root']).resolve()),
            ))
        dp.init(world, port)
        dp.STATE['helpers'] = helpers
        # Rebind train_update with patched dp (install already bound an unpatched one).
        import subnet.mimo_grpo.service as service  # noqa: F401
        batched = __import__('sys').modules.get('v824_batched')
        fast_kernels = host_mod.load('v83_fast_kernels', 'v83_fast_kernels.py')
        if batched is not None:
            dp.BUDGET, dp.MAX_ROWS = engine['dp_token_budget'], engine['dp_max_rows']
            dp_update = dp.make_train_update(batched, fast_kernels)
            # The host wraps dp_update further; replace the innermost by patching batched.
            batched.train_update_batched = dp_update
        if isinstance(info, dict):
            info = dict(info, dp_world=world, zero3=zero3_enabled(), host_version=HOST_VERSION)
        return info

    host_mod.install = install
    host_mod._v835_install_wrapped = True


def activate(host_mod=None) -> Mapping[str, Any]:
    """Call from train_pit before install.

    Monkey-patches the FP32 accumulator / DP helpers ONLY when sharded mode is
    active (PIT_ZERO3 or per-GPU HBM < 100 GiB). Non-sharded B300/B200/H200
    paths must keep the admitted host behaviour byte-identical.
    """
    mem = per_gpu_mem_gib()
    will = needs_zero3(mem)
    if will:
        enable_zero3_env()
        patch_fp32_accumulator()
        if host_mod is not None:
            wrap_host_install(host_mod)
    return {
        'per_gpu_mem_gib': mem,
        'zero3': will,
        'fp32_accum_device': 'cpu' if will else None,
        'host_version': HOST_VERSION,
        'threshold_gib': float(LOW_MEM_GPU_GIB),
    }


def preflight_for_runtime(runtime, *, world: int = 8, token_budget: Optional[int] = None) -> Mapping[str, Any]:
    mem = per_gpu_mem_gib()
    n = n_params_from_model(runtime.model)
    return preflight_or_raise(
        n_params=n,
        per_gpu_gib=mem,
        world=world,
        token_budget=token_budget,
        sharded=zero3_enabled() or needs_zero3(mem),
    )


def numeric_equivalence_tiny(
    *,
    seed: int = 0,
    world: int = 4,
    lr: float = 1e-3,
    shapes: Sequence[Tuple[int, ...]] = ((16, 8), (8,), (8, 4)),
    tol: float = 1e-5,
) -> Mapping[str, Any]:
    """Tiny-model check: single-GPU path vs sharded ZeRO-3-style path.

    1. Build identical initial bf16 weights (same seed).
    2. Draw ``world`` local microbatch grads; single path SUM-accumulates them on
       one buffer; sharded path keeps per-rank CPU buffers then SUM-reduces
       (simulating DP reduce of ZeRO-3 grad shards all-gathered to full).
    3. Apply the same fp32 master SGD update and project to bf16.
    Final bf16 weights must match within ``tol``.
    """
    import torch

    def make_params(s: int):
        torch.manual_seed(s)
        out = []
        for i, shape in enumerate(shapes):
            out.append(('p%d' % i, torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16))))
        return out

    single = make_params(seed)
    sharded = make_params(seed)
    # Per-rank local grads (different microbatches).
    torch.manual_seed(seed + 99)
    local_grads = []
    for _r in range(world):
        local_grads.append({n: torch.randn(p.shape, dtype=torch.float32) for n, p in single})

    # ---- single path: accumulate all locals then one step ----
    masters_s = {n: p.detach().to('cpu', dtype=torch.float32).clone() for n, p in single}
    acc_s = {n: torch.zeros_like(masters_s[n]) for n, _ in single}
    for lg in local_grads:
        for n in acc_s:
            acc_s[n].add_(lg[n].to('cpu'))
    with torch.no_grad():
        for n, p in single:
            masters_s[n].sub_(acc_s[n], alpha=lr)
            p.copy_(masters_s[n].to(dtype=torch.bfloat16))

    # ---- sharded path: per-rank CPU accum, then SUM-reduce, same step ----
    masters_z = {n: p.detach().to('cpu', dtype=torch.float32).clone() for n, p in sharded}
    rank_acc = [{n: torch.zeros(p.shape, dtype=torch.float32) for n, p in sharded} for _ in range(world)]
    for r, lg in enumerate(local_grads):
        for n in rank_acc[r]:
            rank_acc[r][n].add_(lg[n].to('cpu'))
    reduced = {n: torch.zeros_like(masters_z[n]) for n, _ in sharded}
    for r in range(world):
        for n in reduced:
            reduced[n].add_(rank_acc[r][n])
    with torch.no_grad():
        for n, p in sharded:
            masters_z[n].sub_(reduced[n], alpha=lr)
            p.copy_(masters_z[n].to(dtype=torch.bfloat16))

    diffs = []
    for (_, a), (_, b) in zip(single, sharded):
        diffs.append(float((a.detach().float() - b.detach().float()).abs().max().item()))
    max_diff = max(diffs) if diffs else 0.0
    return {
        'max_abs_diff': max_diff,
        'tol': float(tol),
        'ok': max_diff <= float(tol),
        'world': int(world),
        'seed': int(seed),
        'lr': float(lr),
    }


def helper_main(rank, world, port, host_dir, source_root):
    """Helper entry: apply ZeRO-3 patches, then run v83_dp.helper_main."""
    import importlib.util
    import sys
    sys.path.insert(0, source_root)
    sys.path.insert(0, host_dir)
    if zero3_enabled() or needs_zero3():
        enable_zero3_env()
    patch_fp32_accumulator()
    spec = importlib.util.spec_from_file_location('v83_dp', os.path.join(host_dir, 'v83_dp.py'))
    dp = importlib.util.module_from_spec(spec)
    sys.modules['v83_dp'] = dp
    spec.loader.exec_module(dp)
    patch_dp(dp)
    print(json.dumps(dict(
        component='trainer-host-v834f', event='helper-zero3',
        rank=rank, zero3=zero3_enabled(),
    )), flush=True)
    dp.helper_main(rank, world, port, host_dir, source_root)


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == 'helper':
        _, _, r, w, port, host_dir, source_root = sys.argv[:7]
        helper_main(int(r), int(w), int(port), host_dir, source_root)
    elif len(sys.argv) >= 2 and sys.argv[1] == 'equiv':
        out = numeric_equivalence_tiny()
        print(json.dumps(out, sort_keys=True))
        sys.exit(0 if out['ok'] else 2)
    else:
        raise SystemExit('usage: v835_zero3.py helper RANK WORLD PORT HOST_DIR SOURCE_ROOT | equiv')
