"""ZeRO-2-style sharded FP32 gradient accumulation for SN120 GRPO (v8.3.5 / 64K).

Preserves the v834c accumulation math:

  * each micro-batch contributes ``parameter.grad`` (BF16) added into an FP32 buffer
  * after all local micro-batches, contributions are SUM-reduced across data-parallel ranks
  * one global L2 clip, then the host optimizer sees the same full FP32 inventory

Memory changes only (not training math):

  1. FP32 accumulator buffers are **partitioned across ranks** (each rank holds ~1/W of
     trainable numel). This removes the replicated ~36 GB FP32 term on every GPU.
  2. ``register_post_accumulate_grad_hook`` moves each BF16 ``param.grad`` into the local
     contribution (owned shard stays on GPU; non-owned goes to pinned CPU) and frees the
     BF16 grad immediately, so the full ~18 GB BF16 grad inventory never coexists with the
     accumulator.
  3. After local passes, ranks SUM-reduce each tensor onto its owner (one tensor at a time).
     Rank 0 then gathers the owned shards into a full ``buffers`` dict for clip + optimizer,
     matching v834c's rank-0-only step boundary.

Optional activation offload lives in ``v835_dp`` (checkpoint input to CPU), not here.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


NamedParams = Sequence[Tuple[str, torch.nn.Parameter]]


def partition_names(named_parameters: NamedParams, world: int) -> List[List[str]]:
    """Greedy LPT partition of trainable tensors by numel across ``world`` ranks."""
    if world < 1:
        raise ValueError('world must be >= 1')
    items = [(name, int(p.numel())) for name, p in named_parameters if p.requires_grad]
    if not items:
        raise ValueError('unique nonempty trainable parameter inventory')
    if len({n for n, _ in items}) != len(items):
        raise ValueError('duplicate trainable parameter name')
    loads = [0] * world
    owned: List[List[str]] = [[] for _ in range(world)]
    for name, numel in sorted(items, key=lambda it: (-it[1], it[0])):
        r = min(range(world), key=lambda k: (loads[k], k))
        owned[r].append(name)
        loads[r] += numel
    return owned


def ownership_map(named_parameters: NamedParams, world: int) -> Dict[str, int]:
    owned = partition_names(named_parameters, world)
    return {name: rank for rank, names in enumerate(owned) for name in names}


class ShardedFP32Accumulator:
    """FP32 gradient accumulator with ZeRO-2 ownership and post-accumulate disposal."""

    def __init__(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]], *,
                 rank: int = 0, world: int = 1, device: Optional[torch.device] = None):
        params = tuple((name, p) for name, p in named_parameters if p.requires_grad)
        if not params or len({name for name, _ in params}) != len(params):
            raise ValueError('unique nonempty trainable parameter inventory')
        if len({id(p) for _, p in params}) != len(params):
            raise ValueError('duplicate trainable parameter object')
        self.parameters = params
        self.rank = int(rank)
        self.world = int(world)
        if not (0 <= self.rank < self.world):
            raise ValueError('rank out of range')
        self.owners = ownership_map(params, self.world)
        self.owned_names = tuple(name for name, _ in params if self.owners[name] == self.rank)
        if device is None:
            device = params[0][1].device
        self.device = torch.device(device)
        # Owned shard: resident FP32 on the parameter device (GPU in production).
        self.buffers: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, dtype=torch.float32, device=self.device)
            for name, p in params if self.owners[name] == self.rank
        }
        # Non-owned local contributions: pinned CPU FP32 (not HBM).
        pin = self.device.type == 'cuda'
        self._cpu_contrib: Dict[str, torch.Tensor] = {
            name: torch.zeros(p.shape, dtype=torch.float32, device='cpu', pin_memory=pin)
            for name, p in params if self.owners[name] != self.rank
        }
        self.microsteps = 0
        self.clipped = False
        self._hooks = []
        self._by_id = {id(p): name for name, p in params}
        self._param_by_name = {name: p for name, p in params}
        self._hook_mode = False
        self._seen = set()

    def install_hooks(self):
        """Free BF16 ``param.grad`` into the sharded accumulator as soon as it is final."""
        if self._hooks:
            return
        if not hasattr(torch.nn.Parameter, 'register_post_accumulate_grad_hook'):
            raise RuntimeError('torch Parameter.register_post_accumulate_grad_hook is required')

        def make_hook(parameter: torch.nn.Parameter):
            def hook(param: torch.nn.Parameter):
                if self.clipped:
                    raise ValueError('gradient accumulator already clipped')
                name = self._by_id[id(param)]
                grad = param.grad
                if grad is None or grad.is_sparse or grad.shape != param.shape:
                    raise ValueError('missing/dense gradient inventory: ' + name)
                with torch.no_grad():
                    g = grad.detach()
                    if self.owners[name] == self.rank:
                        # Same cast path as v834c FP32Accumulator.capture: FP32.add_(BF16).
                        self.buffers[name].add_(g)
                    else:
                        self._cpu_contrib[name].add_(g.float())
                    param.grad = None
                self._seen.add(name)
                return None
            return hook

        for _, parameter in self.parameters:
            self._hooks.append(parameter.register_post_accumulate_grad_hook(make_hook(parameter)))
        self._hook_mode = True

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._hook_mode = False

    def capture(self):
        """v834c-compatible capture when hooks are not installed (tests / fallback)."""
        if self.clipped:
            raise ValueError('gradient accumulator already clipped')
        if self._hook_mode:
            # Hooks already folded every finished grad; still require a complete inventory
            # (no lingering BF16 grads) and advance the microstep counter.
            expected = {name for name, _ in self.parameters}
            if self._seen != expected:
                missing = sorted(expected - self._seen)
                raise ValueError('missing/dense gradient inventory: ' + (missing[0] if missing else '?'))
            for name, parameter in self.parameters:
                if parameter.grad is not None:
                    raise ValueError('hook mode left a live BF16 grad: ' + name)
            self._seen.clear()
            self.microsteps += 1
            return
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None or gradient.is_sparse or gradient.shape != parameter.shape:
                raise ValueError('missing/dense gradient inventory: ' + name)
        with torch.no_grad():
            for name, parameter in self.parameters:
                g = parameter.grad.detach()
                if self.owners[name] == self.rank:
                    self.buffers[name].add_(g)
                else:
                    self._cpu_contrib[name].add_(g.float())
                parameter.grad = None
        self.microsteps += 1

    def reduce_shards(self, group=None):
        """SUM-reduce each tensor onto its owner. All ranks must call in lockstep."""
        if self.clipped or not self.microsteps:
            raise ValueError('reduce after nonempty accumulation')
        if self.world == 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('torch.distributed is required for world>1 reduce_shards')
        for name, parameter in self.parameters:
            owner = self.owners[name]
            if self.rank == owner:
                dist.reduce(self.buffers[name], dst=owner, op=dist.ReduceOp.SUM, group=group)
            else:
                tmp = self._cpu_contrib[name].to(device=self.device, dtype=torch.float32, non_blocking=False)
                dist.reduce(tmp, dst=owner, op=dist.ReduceOp.SUM, group=group)
                del tmp
                # Drop host contribution after a successful reduce (owner holds the SUM).
                self._cpu_contrib[name].zero_()

    def gather_full_buffers(self, dst: int = 0, group=None) -> Optional[Dict[str, torch.Tensor]]:
        """Materialize the full FP32 inventory on ``dst`` (rank 0 for the host optimizer)."""
        if self.world == 1:
            # Single-process: owned buffers are already the full inventory; fold CPU if any.
            full = {}
            for name, parameter in self.parameters:
                if name in self.buffers:
                    full[name] = self.buffers[name]
                else:
                    full[name] = self._cpu_contrib[name].to(device=self.device)
            return full if self.rank == dst else None
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('torch.distributed is required for world>1 gather_full_buffers')
        full: Dict[str, torch.Tensor] = {}
        for name, parameter in self.parameters:
            owner = self.owners[name]
            if self.rank == owner:
                tensor = self.buffers[name]
            else:
                tensor = torch.empty(parameter.shape, dtype=torch.float32, device=self.device)
            dist.broadcast(tensor, src=owner, group=group)
            if self.rank == dst:
                full[name] = tensor
            elif self.rank != owner:
                del tensor
        return full if self.rank == dst else None

    def local_sqnorm(self) -> torch.Tensor:
        """Sum of squared L2 norms over tensors owned by this rank (FP32)."""
        if not self.buffers:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for gradient in self.buffers.values():
            total = total + torch.linalg.vector_norm(gradient, dtype=torch.float32).square()
        return total

    def clip(self, maximum: float, *, buffers: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """Clip either the gathered full inventory (rank 0) or owned shards (with all-reduce)."""
        if self.clipped or not self.microsteps:
            raise ValueError('one clip after nonempty accumulation')
        target = buffers if buffers is not None else self.buffers
        if buffers is not None:
            norms = [torch.linalg.vector_norm(g, dtype=torch.float32) for g in target.values()]
            total = torch.linalg.vector_norm(torch.stack(norms), dtype=torch.float32)
        else:
            local = self.local_sqnorm()
            if self.world > 1:
                dist.all_reduce(local, op=dist.ReduceOp.SUM)
            total = torch.sqrt(local)
        if not torch.isfinite(total):
            raise ValueError('nonfinite accumulated gradient norm')
        scale = torch.clamp(maximum / (total + 1e-6), max=1.)
        with torch.no_grad():
            for gradient in target.values():
                gradient.mul_(scale)
            if buffers is None:
                pass
            elif self.rank == 0:
                # Keep owned shard views consistent when full dict aliases owned tensors.
                for name, gradient in self.buffers.items():
                    if name in target and target[name] is not gradient:
                        gradient.copy_(target[name])
        self.clipped = True
        return float(total)

    def clear(self):
        self.remove_hooks()
        self.buffers.clear()
        self._cpu_contrib.clear()
        self.microsteps = 0


def estimate_bytes(named_parameters: NamedParams, world: int = 8) -> dict:
    """Static per-rank bytes for the v834c vs v835 layouts (excludes activations/workspaces)."""
    params = [(n, p) for n, p in named_parameters if p.requires_grad]
    total_numel = sum(p.numel() for _, p in params)
    owned = partition_names(params, world)
    sizes = {n: int(p.numel()) for n, p in params}
    shard_numel = [sum(sizes[name] for name in names) for names in owned]
    peak_param = max((p.numel() for _, p in params), default=0)
    bf16_w = total_numel * 2
    return dict(
        trainable_numel=total_numel,
        world=world,
        v834c=dict(
            bf16_weights_bytes=bf16_w,
            fp32_accumulator_bytes=total_numel * 4,
            bf16_param_grad_peak_bytes=bf16_w,
            resident_grad_related_bytes=total_numel * 4 + bf16_w,
        ),
        v835=dict(
            bf16_weights_bytes=bf16_w,
            fp32_accumulator_shard_bytes=max(shard_numel) * 4 if shard_numel else 0,
            bf16_param_grad_peak_bytes=peak_param * 2,  # one tensor in flight under hooks
            reduce_scratch_bytes=peak_param * 4,
            resident_grad_related_bytes=(max(shard_numel) * 4 if shard_numel else 0) + peak_param * 2,
        ),
        shard_numel=shard_numel,
        largest_tensor_numel=peak_param,
        saved_vs_v834c_bytes=(total_numel * 4 + bf16_w) - (
            (max(shard_numel) * 4 if shard_numel else 0) + peak_param * 2
        ),
    )


def gb(n: int) -> float:
    return n / (1024 ** 3)
