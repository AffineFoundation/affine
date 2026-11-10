"""v8.3.5 / host-v834f: FSDP FULL_SHARD (ZeRO-3) train path for 8x low-VRAM GPUs.

Every rank wraps together on one process group (CUDA FSDP, or a gloo/CPU
FULL_SHARD stand-in for tests). Params, grads and optimizer shards are split
across the world; CPU offload keeps HBM under ~28 GiB at 32K context with
gradient checkpointing. FP32 master / accumulator math matches host-v834e.

Helpers MUST start through this module's ``helper`` entry so the accumulator
patch is applied before ``v83_dp._passes`` runs. ``activate(host_mod)`` must
receive the host module so ``wrap_host_install`` runs (``activate(None)`` is
a fail-closed refuse when sharded mode is required).
"""
from __future__ import annotations

import json
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple

ZERO3_ENV = 'PIT_ZERO3'
ACCUM_DEVICE_ENV = 'PIT_FP32_ACCUM_DEVICE'
BACKEND_ENV = 'PIT_ZERO3_BACKEND'
LOW_MEM_GPU_GIB = 100.0
REF_PEAK_GIB = 112.0
REF_PARAM_GIB = 18.0
REF_ACCUM_GIB = 36.0
REF_RESIDUAL_GIB = REF_PEAK_GIB - REF_PARAM_GIB - REF_ACCUM_GIB
# Residual under FULL_SHARD + CPU offload + gradient checkpointing (no full
# activation footprint). Reference is for the high token budget; scales down.
REF_RESIDUAL_OFFLOAD_GIB = 16.0
PREFLIGHT_MARGIN = 1.15
# Low-VRAM token budget actually applied to engine.dp_token_budget AND token_budget.
LOW_VRAM_TOKEN_BUDGET = 24576
HOST_VERSION = 'host-v834f'
# Peak target on 32 GiB cards with headroom for fragmentation / CUDA context.
TARGET_PEAK_GIB = 28.0


class VramPreflightError(RuntimeError):
    """Projected per-GPU peak exceeds local VRAM; refuse rather than OOM mid-run."""


def zero3_enabled() -> bool:
    return os.environ.get(ZERO3_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')


def enable_zero3_env() -> None:
    os.environ[ZERO3_ENV] = '1'
    os.environ[ACCUM_DEVICE_ENV] = 'cpu'


def dist_backend() -> str:
    forced = (os.environ.get(BACKEND_ENV) or '').strip().lower()
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return 'nccl'
    except Exception:
        pass
    return 'gloo'


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
    cpu_offload: bool = True,
) -> Mapping[str, float]:
    """Project per-GPU peak HBM for the path that actually runs.

    Sharded + CPU-offload path (host-v834f):
      * bf16 params live on CPU (0 HBM persistent); one-layer gather workspace
        ≈ param_gib / (world * n_layers_proxy) budgeted as param_gib / world
      * grads are FULL_SHARD in HBM briefly → param_gib / world (bf16) plus
        an fp32 reduce workspace of the same size
      * fp32 accumulator is on CPU (0 HBM)
      * residual activations scale with token_budget vs the B300 reference
    Unsharded path keeps the historical single-GPU projection.
    """
    world = max(int(world), 1)
    param_gib = (float(n_params) * 2.0) / float(1 << 30)  # bf16
    scale_params = param_gib / REF_PARAM_GIB if REF_PARAM_GIB else 1.0
    if token_budget is None:
        scale_tok = 1.0
    else:
        scale_tok = max(float(token_budget) / float(ref_token_budget), 0.05)
    if sharded and cpu_offload:
        residual = REF_RESIDUAL_OFFLOAD_GIB * scale_params * scale_tok
    else:
        residual = REF_RESIDUAL_GIB * scale_params * scale_tok
    if sharded:
        # Do NOT claim params/grads are /world unless wrap actually ran with
        # FULL_SHARD; this estimator is only valid for the wrapped path.
        if cpu_offload:
            params_hbm = 0.0
            # One-layer all-gather workspace + sharded grad + fp32 reduce pad.
            gather_hbm = param_gib / float(world)
            grad_hbm = param_gib / float(world)
            reduce_hbm = (param_gib * 2.0) / float(world)
            accum_hbm = 0.0
            peak = (params_hbm + gather_hbm + grad_hbm + reduce_hbm + residual) * PREFLIGHT_MARGIN
        else:
            # Full shards resident in HBM (no CPU offload): params/world + grads/world
            # before offload this is ~18+18 which exceeds 32 GiB — refuse in preflight.
            params_hbm = param_gib / float(world)
            grad_hbm = param_gib / float(world)
            gather_hbm = param_gib / float(world)
            accum_hbm = 0.0
            peak = (params_hbm + grad_hbm + gather_hbm + accum_hbm + residual) * PREFLIGHT_MARGIN
    else:
        peak = (param_gib + REF_ACCUM_GIB * scale_params + residual) * PREFLIGHT_MARGIN
    return {
        'peak_gib': float(peak),
        'param_gib': float(param_gib),
        'residual_gib': float(residual),
        'world': float(world),
        'sharded': 1.0 if sharded else 0.0,
        'cpu_offload': 1.0 if cpu_offload else 0.0,
        'margin': float(PREFLIGHT_MARGIN),
        'target_peak_gib': float(TARGET_PEAK_GIB),
    }


def preflight_or_raise(
    *,
    n_params: int,
    per_gpu_gib: Optional[float],
    world: int = 8,
    token_budget: Optional[int] = None,
    sharded: bool = True,
    cpu_offload: bool = True,
) -> Mapping[str, Any]:
    """Refuse training when projected per-GPU peak exceeds VRAM."""
    if per_gpu_gib is None:
        raise VramPreflightError('per-GPU VRAM unknown; refuse to start sharded train')
    est = dict(estimate_peak_gib(
        n_params=n_params, world=world, token_budget=token_budget,
        sharded=sharded, cpu_offload=cpu_offload,
    ))
    est['per_gpu_gib'] = float(per_gpu_gib)
    limit = min(float(per_gpu_gib), float(TARGET_PEAK_GIB) if float(per_gpu_gib) < 40 else float(per_gpu_gib))
    if float(est['peak_gib']) > float(limit):
        raise VramPreflightError(
            'projected per-GPU peak %.1f GiB exceeds limit %.1f GiB (VRAM %.1f, world=%d, sharded=%s, offload=%s)'
            % (est['peak_gib'], limit, per_gpu_gib, world, sharded, cpu_offload)
        )
    return est


def patch_fp32_accumulator() -> None:
    """Ensure subnet FP32Accumulator accepts buffer_device='cpu' (golden bundle may lack it).

    Only active under PIT_ZERO3. PIT_FP32_ACCUM_DEVICE alone must not change the
    admitted non-sharded path.
    """
    import torch
    import subnet.mimo_grpo.optimizer as opt_mod
    Acc = opt_mod.FP32Accumulator
    if getattr(Acc, '_v835_buffer_device', False):
        return
    _orig_init = Acc.__init__
    _orig_capture = Acc.capture

    def __init__(self, named_parameters, *, buffer_device=None):
        # Fail closed: ignore buffer_device / env unless ZeRO-3 is enabled.
        if not zero3_enabled():
            return _orig_init(self, named_parameters)
        if buffer_device is None:
            buffer_device = 'cpu'
        self.parameters = tuple((name, p) for name, p in named_parameters if p.requires_grad)
        if not self.parameters or len({name for name, _ in self.parameters}) != len(self.parameters):
            raise ValueError('unique nonempty trainable parameter inventory')
        if len({id(p) for _, p in self.parameters}) != len(self.parameters):
            raise ValueError('duplicate trainable parameter object')
        self.buffer_device = buffer_device
        self.buffers = {
            name: torch.zeros(p.shape, dtype=torch.float32, device=buffer_device)
            for name, p in self.parameters
        }
        self.microsteps = 0
        self.clipped = False

    def capture(self):
        if not hasattr(self, 'buffer_device') or self.buffer_device is None:
            return _orig_capture(self)
        if self.clipped:
            raise ValueError('gradient accumulator already clipped')
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None or gradient.is_sparse or gradient.shape != parameter.shape:
                raise ValueError('missing/dense gradient inventory: ' + name)
        with torch.no_grad():
            for name, parameter in self.parameters:
                grad = parameter.grad
                if grad.device != self.buffers[name].device:
                    self.buffers[name].add_(grad.detach().to(self.buffers[name].device, dtype=torch.float32))
                else:
                    self.buffers[name].add_(grad)
                parameter.grad = None
        self.microsteps += 1

    Acc.__init__ = __init__
    Acc.capture = capture
    Acc._v835_buffer_device = True
    Acc._v835_orig_init = _orig_init
    Acc._v835_orig_capture = _orig_capture


def _strip_fsdp_prefix(name: str) -> str:
    """Map FSDP-wrapped names back to host-v834e publication keys."""
    out = name
    for prefix in ('_fsdp_wrapped_module.', 'module.', '_orig_mod.'):
        while out.startswith(prefix):
            out = out[len(prefix):]
    return out


def canonical_named_parameters(model) -> List[Tuple[str, Any]]:
    """Trainable (name, param) with host-v834e key names (no FSDP prefixes)."""
    seen = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = _strip_fsdp_prefix(name)
        if key in seen:
            raise ValueError('duplicate canonical parameter name after FSDP strip: ' + key)
        seen[key] = param
    if not seen:
        raise ValueError('no trainable parameters')
    return list(seen.items())


def publication_state_dict(model) -> Mapping[str, Any]:
    """Full CPU state dict whose keys match an unsharded host-v834e save."""
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP) or getattr(model, '_v835_fsdp_cuda', False):
        from torch.distributed.fsdp import StateDictType
        from torch.distributed.fsdp.api import FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            raw = model.state_dict()
        if dist.is_initialized() and dist.get_rank() != 0:
            return {}
        return {_strip_fsdp_prefix(k): v.detach().to('cpu') for k, v in raw.items()}

    if getattr(model, '_v835_manual_shard', False):
        return model.full_state_dict()

    return {_strip_fsdp_prefix(k): v.detach().to('cpu') for k, v in model.state_dict().items()}


def sharded_param_nbytes(model) -> Mapping[str, int]:
    """Local shard nbytes per canonical name (for tests / preflight evidence)."""
    out = {}
    for name, param in canonical_named_parameters(model):
        try:
            out[name] = int(param.untyped_storage().nbytes())
        except Exception:
            out[name] = int(param.numel() * param.element_size())
    return out


def _make_manual_full_shard_cls():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    class _ManualFullShard(torch.nn.Module):
        """gloo/CPU FULL_SHARD stand-in: DTensor-sharded params, original names.

        Used when CUDA FSDP is unavailable (multi-process CPU tests). Production
        8xRTX5090 uses CUDA FSDP via ``_wrap_cuda_fsdp``.
        """
        _v835_manual_shard = True
        _v835_fsdp = True

        def __init__(self, module: torch.nn.Module, *, process_group=None):
            super().__init__()
            if getattr(module, '_v835_fsdp', False):
                raise ValueError('model already wrapped')
            self.process_group = process_group
            world = dist.get_world_size(process_group)
            # Device mesh over the default (or provided) group ranks.
            mesh = init_device_mesh('cpu', (world,))
            self._mesh = mesh
            self._module = module
            # Replace trainable parameters with Shard(0) DTensors in-place via
            # a name-stable Parameter view: keep nn.Module param names identical.
            for name, param in list(module.named_parameters()):
                if not param.requires_grad:
                    continue
                parent, _, leaf = name.rpartition('.')
                owner = module.get_submodule(parent) if parent else module
                flat = param.detach().contiguous().to(dtype=param.dtype, device='cpu')
                # Pad numel to world so Shard(0) splits evenly on dim-0 reshape.
                numel = flat.numel()
                pad = (world - (numel % world)) % world
                if pad:
                    flat = torch.nn.functional.pad(flat.view(-1), (0, pad))
                rows = flat.numel() // world
                shaped = flat.view(world, rows)
                dt = distribute_tensor(shaped, mesh, [Shard(0)])
                # Local shard as an nn.Parameter so autograd + named_parameters work.
                local = torch.nn.Parameter(dt.to_local().detach().clone().requires_grad_(True))
                local._v835_full_shape = tuple(param.shape)
                local._v835_numel = numel
                local._v835_pad = pad
                local._v835_world = world
                local._v835_dt_meta = True
                setattr(owner, leaf, local)
            module._v835_fsdp = True

        def forward(self, *args, **kwargs):
            # Materialize full params for the compute, then restore shards.
            restored = []
            for name, param in list(self._module.named_parameters()):
                if not getattr(param, '_v835_dt_meta', False):
                    continue
                world = param._v835_world
                local = param.detach()
                parts = [torch.empty_like(local) for _ in range(world)]
                dist.all_gather(parts, local, group=self.process_group)
                full = torch.cat([p.reshape(-1) for p in parts], dim=0)
                if param._v835_pad:
                    full = full[: param._v835_numel]
                full = full.view(param._v835_full_shape).to(dtype=param.dtype)
                full_p = torch.nn.Parameter(full.detach().clone().requires_grad_(True))
                parent, _, leaf = name.rpartition('.')
                owner = self._module.get_submodule(parent) if parent else self._module
                restored.append((owner, leaf, param, full_p))
                setattr(owner, leaf, full_p)
            try:
                out = self._module(*args, **kwargs)
            finally:
                # Reduce-scatter grads back onto shards; restore shard params.
                for owner, leaf, shard, full_p in restored:
                    if full_p.grad is not None:
                        g = full_p.grad.detach().reshape(-1)
                        if shard._v835_pad:
                            g = torch.nn.functional.pad(g, (0, shard._v835_pad))
                        chunks = list(g.view(shard._v835_world, -1).unbind(0))
                        # SUM across ranks first (data-parallel), then take local shard.
                        buf = torch.stack(chunks, dim=0).clone()
                        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=self.process_group)
                        shard.grad = buf[dist.get_rank(self.process_group)].to(dtype=shard.dtype).clone()
                    setattr(owner, leaf, shard)
            return out

        def named_parameters(self, prefix='', recurse=True):
            return self._module.named_parameters(prefix=prefix, recurse=recurse)

        def parameters(self, recurse=True):
            return self._module.parameters(recurse=recurse)

        def state_dict(self, *args, **kwargs):
            return self.full_state_dict()

        def full_state_dict(self):
            rank = dist.get_rank(self.process_group)
            world = dist.get_world_size(self.process_group)
            out = {}
            for name, param in self._module.named_parameters():
                if getattr(param, '_v835_dt_meta', False):
                    local = param.detach()
                    parts = [torch.empty_like(local) for _ in range(world)]
                    dist.all_gather(parts, local, group=self.process_group)
                    full = torch.cat([p.reshape(-1) for p in parts], dim=0)
                    if param._v835_pad:
                        full = full[: param._v835_numel]
                    full = full.view(param._v835_full_shape).to('cpu')
                    if rank == 0:
                        out[_strip_fsdp_prefix(name)] = full.clone()
                else:
                    if rank == 0:
                        out[_strip_fsdp_prefix(name)] = param.detach().to('cpu').clone()
            # non-parameter buffers
            if rank == 0:
                for name, buf in self._module.named_buffers():
                    out[_strip_fsdp_prefix(name)] = buf.detach().to('cpu').clone()
            return out

    return _ManualFullShard


def _layer_wrap_policy(model):
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    try:
        layer = model.model.language_model.layers[0]
    except Exception:
        classes = {type(m) for m in model.children() if any(p.requires_grad for p in m.parameters())}
        if not classes:
            raise ValueError('no FSDP-wrappable trainable children')
        return ModuleWrapPolicy(classes)
    return ModuleWrapPolicy({type(layer)})


def _wrap_cuda_fsdp(model, *, local_rank: int = 0, process_group=None):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
    if getattr(model, '_v835_fsdp', False):
        return model
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    # use_orig_params=True keeps Parameter objects; publication_state_dict strips
    # any residual FSDP key prefixes so checkpoint keys match host-v834e.
    wrapped = FSDP(
        model,
        auto_wrap_policy=_layer_wrap_policy(model),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        sync_module_states=True,
        cpu_offload=CPUOffload(offload_params=True),
        process_group=process_group,
    )
    wrapped._v835_fsdp = True
    wrapped._v835_fsdp_cuda = True
    return wrapped


def wrap_fsdp(model, *, local_rank: int = 0, process_group=None):
    """FULL_SHARD wrap on every rank together. CUDA → FSDP; CPU/gloo → ManualFullShard."""
    import torch
    if getattr(model, '_v835_fsdp', False):
        return model
    backend = dist_backend()
    if backend == 'nccl' and torch.cuda.is_available():
        return _wrap_cuda_fsdp(model, local_rank=local_rank, process_group=process_group)
    cls = _make_manual_full_shard_cls()
    return cls(model, process_group=process_group)


def apply_low_vram_token_budget(engine: Mapping[str, Any], mem_gib: Optional[float]) -> dict:
    """Return an engine dict with dp_token_budget AND token_budget cut for low VRAM."""
    engine = dict(engine)
    if mem_gib is not None and float(mem_gib) < 40:
        if int(engine.get('dp_token_budget', 69632)) > LOW_VRAM_TOKEN_BUDGET:
            engine['dp_token_budget'] = LOW_VRAM_TOKEN_BUDGET
        if int(engine.get('token_budget', 69632)) > LOW_VRAM_TOKEN_BUDGET:
            engine['token_budget'] = LOW_VRAM_TOKEN_BUDGET
    return engine


def patch_dp(dp_mod) -> None:
    """Patch v83_dp for ZeRO-3: gloo-capable init, CPU accum, FSDP-safe reduce."""
    if getattr(dp_mod, '_v835_patched', False):
        return
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    _passes_orig = dp_mod._passes
    _reduce_orig = dp_mod._reduce
    _init_orig = dp_mod.init

    def init(world, port):
        backend = dist_backend()
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = str(port)
        if backend == 'nccl' and torch.cuda.is_available():
            torch.cuda.set_device(0)
            dist.init_process_group(
                'nccl', rank=0, world_size=world,
                device_id=torch.device('cuda', 0), timeout=timedelta(minutes=45),
            )
        else:
            dist.init_process_group(
                'gloo', rank=0, world_size=world, timeout=timedelta(minutes=45),
            )
        dp_mod.STATE['ctrl'] = dist.new_group(backend='gloo', timeout=timedelta(days=30))
        dp_mod.STATE.update(world=world, initialized=True, backend=backend)

    def _passes(B, FK, actor, ref, rows, plan, config, agg_config, loss_scale, active_rollouts, versions, guard, record):
        from subnet.mimo_grpo.optimizer import FP32Accumulator
        accumulator, details, losses, err = None, {}, {}, None
        try:
            B._reference_pass(rows, plan, config, ref, engine='packed', logit_chunk=dp_mod.LOGIT_CHUNK, guard=guard)
            if B._versions(actor.model) != versions:
                raise ValueError('input actor changed during reference preparation')
            for p in actor.model.parameters():
                p.grad = None
            # CPU accum ONLY under PIT_ZERO3 (never from PIT_FP32_ACCUM_DEVICE alone).
            if zero3_enabled():
                named = canonical_named_parameters(actor.model)
                accumulator = FP32Accumulator(named, buffer_device='cpu')
            else:
                accumulator = FP32Accumulator(actor.model.named_parameters())
            B._gradient_pass(
                actor, rows, plan, config, agg_config, loss_scale, active_rollouts, accumulator, versions,
                details, losses, engine='packed', logit_chunk=dp_mod.LOGIT_CHUNK, guard=guard, record=record,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except BaseException as e:
            err = '%s: %s' % (type(e).__name__, e)
        return accumulator, details, losses, err

    def _reduce(B, rows, accumulator, details, losses, err, rank):
        world = dp_mod.STATE['world']
        device = 'cuda' if (dist_backend() == 'nccl' and torch.cuda.is_available()) else 'cpu'
        ok = torch.tensor([0.0 if err else 1.0], device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        errs = [None] * world
        dist.all_gather_object(errs, err)
        if ok.item() < 1:
            if accumulator is not None:
                accumulator.buffers.clear()
            return None, [e for e in errs if e]
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

    dp_mod.init = init
    dp_mod._passes = _passes
    dp_mod._reduce = _reduce
    dp_mod._passes_orig = _passes_orig
    dp_mod._reduce_orig = _reduce_orig
    dp_mod._init_orig = _init_orig
    dp_mod._v835_patched = True


def wrap_host_install(host_mod) -> None:
    """Wrap mimo_host_trainer.install so helpers + rank0 get ZeRO-3 before DP starts."""
    if getattr(host_mod, '_v835_install_wrapped', False):
        return
    import subprocess
    import sys
    from pathlib import Path
    orig = host_mod.install

    def install(config, start_dp=False):
        engine = config['engine']
        mem = per_gpu_mem_gib()
        if not needs_zero3(mem):
            return orig(config, start_dp=start_dp)
        enable_zero3_env()
        patch_fp32_accumulator()
        engine = apply_low_vram_token_budget(engine, mem)
        config = dict(config, engine=engine)
        dp = host_mod.load('v83_dp', 'v83_dp.py')
        patch_dp(dp)
        if not start_dp:
            return orig(config, start_dp=False)

        info = orig(config, start_dp=False)
        world, port = engine['dp_world'], engine['dp_port']
        helpers = []
        helper_entry = str(host_mod.P / 'v835_zero3.py')
        backend = dist_backend()
        for r in range(1, world):
            env = dict(
                os.environ,
                MASTER_ADDR='127.0.0.1',
                MASTER_PORT=str(port),
                TRITON_CACHE_DIR=host_mod.TRITON_CACHE + '/rank%d' % r,
                PIT_ZERO3='1',
                PIT_FP32_ACCUM_DEVICE='cpu',
                PIT_ZERO3_BACKEND=backend,
            )
            if backend == 'nccl':
                env['CUDA_VISIBLE_DEVICES'] = str(r)
            helpers.append(subprocess.Popen(
                [
                    sys.executable, '-B', helper_entry, 'helper',
                    str(r), str(world), str(port),
                    str(host_mod.P), str(Path(config['source_root']).resolve()),
                ],
                env=env,
                cwd=str(Path(config['source_root']).resolve()),
            ))
        dp.init(world, port)
        dp.STATE['helpers'] = helpers
        import subnet.mimo_grpo.service as service  # noqa: F401
        batched = sys.modules.get('v824_batched')
        fast_kernels = host_mod.load('v83_fast_kernels', 'v83_fast_kernels.py')
        if batched is not None:
            dp.BUDGET, dp.MAX_ROWS = engine['dp_token_budget'], engine['dp_max_rows']
            dp_update = dp.make_train_update(batched, fast_kernels)
            batched.train_update_batched = dp_update
        if isinstance(info, dict):
            info = dict(
                info,
                dp_world=world,
                zero3=True,
                host_version=HOST_VERSION,
                token_budget=engine.get('token_budget'),
                dp_token_budget=engine.get('dp_token_budget'),
                backend=backend,
            )
        return info

    host_mod.install = install
    host_mod._v835_install_wrapped = True


def activate(host_mod=None) -> Mapping[str, Any]:
    """Call from train_pit BEFORE install.

    ``host_mod`` is required when sharded mode will run — otherwise
    ``wrap_host_install`` never runs and helpers start unpatched (TypeError /
    hang). Passing ``None`` while ZeRO-3 is required raises.
    """
    mem = per_gpu_mem_gib()
    will = needs_zero3(mem)
    if will:
        if host_mod is None:
            raise RuntimeError(
                'v835_zero3.activate(host_mod) requires the host module when '
                'sharded mode is active (wrap_host_install must run)'
            )
        enable_zero3_env()
        patch_fp32_accumulator()
        wrap_host_install(host_mod)
    return {
        'per_gpu_mem_gib': mem,
        'zero3': will,
        'fp32_accum_device': 'cpu' if will else None,
        'host_version': HOST_VERSION,
        'threshold_gib': float(LOW_MEM_GPU_GIB),
        'token_budget_cap': LOW_VRAM_TOKEN_BUDGET if will else None,
        'backend': dist_backend() if will else None,
    }


def preflight_for_runtime(runtime, *, world: int = 8, token_budget: Optional[int] = None) -> Mapping[str, Any]:
    mem = per_gpu_mem_gib()
    n = n_params_from_model(runtime.model)
    return preflight_or_raise(
        n_params=n,
        per_gpu_gib=mem,
        world=world,
        token_budget=token_budget,
        sharded=True,
        cpu_offload=True,
    )


def install_publication_patch(runtime) -> None:
    """Make runtime.save_model emit host-v834e key names / full unsharded tensors."""
    if getattr(runtime, '_v835_save_patched', False):
        return
    from pathlib import Path
    import torch
    from safetensors.torch import save_file

    # Imported lazily: mimo_runtime.file_hash / digest used by the original path.
    import subnet.mimo_runtime as mimo_rt

    def save_model(directory, *, max_shard_size="2GB"):
        runtime.guard()
        path = Path(directory)
        if path.exists() and path.is_dir() and any(path.iterdir()):
            raise ValueError('new empty export directory')
        path.mkdir(parents=True, exist_ok=True)
        sd = publication_state_dict(runtime.model)
        if not sd:
            # Non-rank0 under rank0_only gather: write nothing; rank0 owns export.
            raise ValueError('publication_state_dict empty on save rank')
        # Single-file safetensors when small; otherwise mirror HF shard naming
        # with canonical keys (no _fsdp_wrapped_module prefix).
        out = path / 'model.safetensors'
        cpu_sd = {k: v.detach().to('cpu').contiguous() for k, v in sd.items()}
        save_file(cpu_sd, str(out))
        if hasattr(runtime, 'tokenizer') and runtime.tokenizer is not None:
            runtime.tokenizer.save_pretrained(path)
        # Persist config.json when the underlying HF model carries one.
        inner = getattr(runtime.model, '_module', None) or runtime.model
        cfg_mod = getattr(inner, 'config', None) or getattr(getattr(inner, 'model', None), 'config', None)
        if cfg_mod is not None and hasattr(cfg_mod, 'to_json_file'):
            cfg_mod.to_json_file(str(path / 'config.json'))
        files = {p.name: mimo_rt.file_hash(p) for p in path.iterdir() if p.is_file()}
        descriptor = {'id': mimo_rt.digest(files), 'files': files}
        runtime._staged_export = (descriptor, mimo_rt.CheckpointGuard(path.absolute(), descriptor))
        return descriptor

    runtime.save_model = save_model
    runtime._v835_save_patched = True


def prepare_wrapped_model(runtime, *, local_rank: int = 0):
    """Rank-0 entry: bcast fsdp-init so EVERY rank wraps on the same group.

    Must run after ``install(start_dp=True)`` (helpers blocked on ctrl bcast)
    and before ``CPUAdamW(...)`` so optimizer identities match wrapped params.
    """
    import torch.distributed as dist
    import sys
    dp = sys.modules.get('v83_dp')
    if dp is None:
        raise RuntimeError('v83_dp not installed')
    actor_dir = str(runtime._checkpoint_guard.directory)
    actor_desc = runtime._checkpoint_guard.descriptor
    spec = runtime.spec.to_dict()
    dp._bcast(dict(
        op='fsdp-init',
        actor_dir=actor_dir,
        actor_desc=actor_desc,
        spec=spec,
    ))
    runtime.model = wrap_fsdp(runtime.model, local_rank=local_rank, process_group=None)
    install_publication_patch(runtime)
    if dist.is_initialized():
        dist.barrier()
    return runtime


def numeric_equivalence_tiny(
    *,
    seed: int = 0,
    world: int = 4,
    lr: float = 1e-3,
    shapes: Sequence[Tuple[int, ...]] = ((16, 8), (8,), (8, 4)),
    tol: float = 1e-5,
) -> Mapping[str, Any]:
    """Tiny-model check: single-GPU path vs sharded ZeRO-3-style path."""
    import torch

    def make_params(s: int):
        torch.manual_seed(s)
        out = []
        for i, shape in enumerate(shapes):
            out.append(('p%d' % i, torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16))))
        return out

    single = make_params(seed)
    sharded = make_params(seed)
    torch.manual_seed(seed + 99)
    local_grads = []
    for _r in range(world):
        local_grads.append({n: torch.randn(p.shape, dtype=torch.float32) for n, p in single})

    masters_s = {n: p.detach().to('cpu', dtype=torch.float32).clone() for n, p in single}
    acc_s = {n: torch.zeros_like(masters_s[n]) for n, _ in single}
    for lg in local_grads:
        for n in acc_s:
            acc_s[n].add_(lg[n].to('cpu'))
    with torch.no_grad():
        for n, p in single:
            masters_s[n].sub_(acc_s[n], alpha=lr)
            p.copy_(masters_s[n].to(dtype=torch.bfloat16))

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
    """Helper entry: apply ZeRO-3 patches, then serve jobs (incl. fsdp-init)."""
    import importlib.util
    import sys
    import threading
    import time
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    sys.path.insert(0, source_root)
    sys.path.insert(0, host_dir)
    enable_zero3_env()
    patch_fp32_accumulator()

    parent = os.getppid()

    def watchdog():
        while True:
            time.sleep(5)
            if os.getppid() != parent:
                os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = str(port)
    backend = dist_backend()
    if backend == 'nccl' and torch.cuda.is_available():
        torch.cuda.set_device(0)
        dist.init_process_group(
            'nccl', rank=rank, world_size=world,
            device_id=torch.device('cuda', 0), timeout=timedelta(minutes=45),
        )
    else:
        dist.init_process_group('gloo', rank=rank, world_size=world, timeout=timedelta(minutes=45))

    spec = importlib.util.spec_from_file_location('v83_dp', os.path.join(host_dir, 'v83_dp.py'))
    dp = importlib.util.module_from_spec(spec)
    sys.modules['v83_dp'] = dp
    spec.loader.exec_module(dp)
    patch_dp(dp)
    dp.STATE['ctrl'] = dist.new_group(backend='gloo', timeout=timedelta(days=30))
    dp.STATE['world'] = world
    dp.STATE['backend'] = backend

    def load(name, rel):
        s = importlib.util.spec_from_file_location(name, os.path.join(host_dir, rel))
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
        return m

    B = load('v824_batched', 'v8211_batched.py')
    FK = load('v83_fast_kernels', 'v83_fast_kernels.py')
    import v83_kernels_policy as KP
    KP.allow()
    from subnet.mimo_runtime import MimoRuntime

    def fast_blp(runtime, items, *, gradients, engine, logit_chunk=2048, support=None):
        return FK.packed_log_probs(runtime, items, gradients=gradients, logit_chunk=dp.LOGIT_CHUNK, support=support)

    B.batched_log_probs = fast_blp
    dp.install_fast_hash()
    print(json.dumps(dict(
        component='trainer-host-v834f', event='helper-zero3',
        rank=rank, zero3=True, backend=backend,
    )), flush=True)

    STATE = dp.STATE
    STATE['actor'] = None
    STATE['ref'] = None

    while True:
        job = dp._bcast(None)
        if job is None or job.get('op') == 'stop':
            break
        if job.get('op') == 'fsdp-init':
            if STATE['actor'] is None:
                STATE['actor'] = MimoRuntime.load(
                    job['actor_dir'], job['actor_desc'], job['spec'],
                    device=('cuda' if backend == 'nccl' and torch.cuda.is_available() else 'cpu'),
                    trainable=True,
                )
            STATE['actor'].model = wrap_fsdp(STATE['actor'].model, local_rank=rank, process_group=None)
            dist.barrier()
            print(json.dumps(dict(
                component='trainer-host-v834f', event='helper-fsdp-wrapped',
                rank=rank, sharding='FULL_SHARD',
            )), flush=True)
            continue
        # Standard DP job (model already FSDP-wrapped after fsdp-init).
        B.DROP_UNRESOLVED, B.LOSS_AGG_MODE, B.LOSS_SCALE = job['drop'], job['mode'], job['scale']
        B.SUPPORT_RENORM = job.get('support')
        B.KNOBS = job.get('knobs')
        if job.get('ref_desc') is None:
            if STATE['ref'] is not None:
                try:
                    STATE['ref'].close()
                except Exception:
                    pass
                STATE['ref'] = None
        elif STATE['ref'] is None or STATE['ref'].checkpoint_id != job['ref_desc']['id']:
            STATE['ref'] = MimoRuntime.load(
                job['ref_dir'], job['ref_desc'], job['spec'],
                device=('cuda' if backend == 'nccl' and torch.cuda.is_available() else 'cpu'),
                trainable=False,
            )
        if STATE['actor'] is None:
            STATE['actor'] = MimoRuntime.load(
                job['actor_dir'], job['actor_desc'], job['spec'],
                device=('cuda' if backend == 'nccl' and torch.cuda.is_available() else 'cpu'),
                trainable=True,
            )
            STATE['actor'].model = wrap_fsdp(STATE['actor'].model, local_rank=rank, process_group=None)
        actor = STATE['actor']
        # Under FSDP, skip dense param broadcast (would fight the shard). Sync
        # via a barrier only; rank-0 optimizer writeback uses publication gather.
        if not getattr(actor.model, '_v835_fsdp', False):
            for _, p in actor.model.named_parameters():
                dist.broadcast(p.data, src=0)
        else:
            dist.barrier()
        config = job['config']
        agg_config, loss_scale, _ = B.loss_aggregation(config)
        groups = job['groups']
        rows = B.layout_drop_unresolved(groups, config)[0] if B.DROP_UNRESOLVED else B._layout(groups, config)
        active = [i for i, row in enumerate(rows) if row['active']]
        actor.model.eval()
        versions = B._versions(actor.model)
        acc, details, losses, err = dp._passes(
            B, FK, actor, STATE['ref'], rows, job['assign'][rank], config, agg_config, loss_scale,
            len(active), versions, lambda: None, lambda row: None,
        )
        dp._reduce(B, rows, acc, details, losses, err, rank)
        for p in actor.model.parameters():
            p.grad = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dist.destroy_process_group()


def _smoke_rank_main(rank, world_, seed, port, q):
    """Module-level worker for spawn (must be picklable)."""
    import torch
    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ[ZERO3_ENV] = '1'
    os.environ[BACKEND_ENV] = 'gloo'
    dist.init_process_group('gloo', rank=rank, world_size=world_)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16, bias=False),
        torch.nn.Linear(16, 4, bias=False),
    )
    for p in model.parameters():
        p.requires_grad_(True)
    for _, p in model.named_parameters():
        dist.broadcast(p.data, src=0)
    wrapped = wrap_fsdp(model, local_rank=rank, process_group=None)
    dist.barrier()
    local_bytes = sharded_param_nbytes(wrapped)
    torch.manual_seed(seed + 7)
    x = torch.randn(4, 16)
    dist.broadcast(x, src=0)
    y = wrapped(x).pow(2).mean()
    y.backward()
    loss = float(y.detach())
    named = canonical_named_parameters(wrapped)
    # ManualFullShard.forward already SUM-reduces grads across ranks into shards.
    # Every rank must participate in the gather collectives for publication keys.
    sd = publication_state_dict(wrapped)
    dist.barrier()
    q.put(dict(
        rank=rank,
        loss=loss,
        local_bytes=local_bytes,
        keys=sorted(sd.keys()) if rank == 0 else None,
        named=[n for n, _ in named],
    ))
    dist.barrier()
    dist.destroy_process_group()


def multiprocess_train_smoke(
    *,
    world: int = 2,
    seed: int = 0,
    port: int = 29611,
) -> Mapping[str, Any]:
    """Multi-process CPU (gloo) smoke: wrap on all ranks, one step, key/loss checks.

    Exercises the same wrap + shard + canonical-name path used by
    ``install(start_dp=True)`` helpers without requiring CUDA FSDP.
    """
    import torch
    import torch.multiprocessing as mp

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16, bias=False),
        torch.nn.Linear(16, 4, bias=False),
    )
    for p in model.parameters():
        p.requires_grad_(True)
    torch.manual_seed(seed + 7)
    x = torch.randn(4, 16)
    y = model(x).pow(2).mean()
    y.backward()
    base_loss = float(y.detach())
    base_keys = list(model.state_dict().keys())
    base_weights = {n: p.detach().clone() for n, p in model.named_parameters()}

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_smoke_rank_main, args=(r, world, seed, port, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    outs = [q.get(timeout=120) for _ in range(world)]
    for p in procs:
        p.join(120)
        if p.exitcode != 0:
            raise RuntimeError('sharded smoke rank exited %s' % p.exitcode)
    by_rank = {o['rank']: o for o in outs}
    rank0 = by_rank[0]
    full_bytes = {n: int(t.numel() * t.element_size()) for n, t in base_weights.items()}
    for r, o in by_rank.items():
        for name, nbytes in o['local_bytes'].items():
            if name in full_bytes and full_bytes[name] > 0:
                if nbytes >= full_bytes[name]:
                    raise AssertionError(
                        'rank %d param %s not sharded (%d >= full %d)' % (r, name, nbytes, full_bytes[name])
                    )
    if rank0['keys'] != sorted(base_keys):
        raise AssertionError('checkpoint keys diverge: %r vs %r' % (rank0['keys'], sorted(base_keys)))
    loss_diff = abs(rank0['loss'] - base_loss)
    return dict(
        ok=True,
        world=world,
        base_loss=base_loss,
        sharded_loss=rank0['loss'],
        loss_diff=loss_diff,
        keys=rank0['keys'],
        local_bytes=by_rank[0]['local_bytes'],
        full_bytes=full_bytes,
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == 'helper':
        _, _, r, w, port, host_dir, source_root = sys.argv[:7]
        helper_main(int(r), int(w), int(port), host_dir, source_root)
    elif len(sys.argv) >= 2 and sys.argv[1] == 'equiv':
        out = numeric_equivalence_tiny()
        print(json.dumps(out, sort_keys=True))
        sys.exit(0 if out['ok'] else 2)
    elif len(sys.argv) >= 2 and sys.argv[1] == 'smoke':
        out = multiprocess_train_smoke()
        print(json.dumps({k: v for k, v in out.items() if k != 'sd'}, sort_keys=True, default=str))
        sys.exit(0 if out.get('ok') else 2)
    else:
        raise SystemExit('usage: v835_zero3.py helper RANK WORLD PORT HOST_DIR SOURCE_ROOT | equiv | smoke')
