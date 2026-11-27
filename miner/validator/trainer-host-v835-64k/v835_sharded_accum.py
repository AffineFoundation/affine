"""ZeRO-2-style sharded FP32 gradient accumulation for SN120 GRPO (v8.3.5 / 64K).

Preserves the v834c accumulation math:

  * each micro-batch contributes ``parameter.grad`` (BF16) summed in FP32 across ranks
  * after all micro-batches, one global L2 clip
  * rank 0 sees the full FP32 inventory for the host optimizer

Memory / speed (H200-measured prototype ported from ``bench_step.py``):

  1. FP32 accumulator buffers are partitioned across ranks (~1/W of trainable numel).
  2. Post-accumulate hooks cast BF16→FP32 **on GPU** and ``dist.reduce`` (SUM, async) straight
     to the owner on a side CUDA stream (bounded in-flight). No per-rank CPU contrib buffers
     and no separate ``reduce_shards`` pass after backward.
  3. Clip runs on owned GPU shards (all-reduce of sum-of-squares) before gather — identical to
     the v834c global-norm clip.
  4. Rank 0 gathers shards by send/recv into **pinned CPU** tensors (feeds CPU FP32 masters
     without a pageable GPU→host copy of a full 36 GB dict).
"""
from __future__ import annotations

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
    """FP32 gradient accumulator with ZeRO-2 ownership and async GPU reduce-to-owner hooks."""

    def __init__(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]], *,
                 rank: int = 0, world: int = 1, device: Optional[torch.device] = None,
                 max_inflight: int = 4):
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
        if max_inflight < 1:
            raise ValueError('max_inflight must be >= 1')
        self.max_inflight = int(max_inflight)
        self.owners = ownership_map(params, self.world)
        self.owned_names = tuple(name for name, _ in params if self.owners[name] == self.rank)
        if device is None:
            device = params[0][1].device
        self.device = torch.device(device)
        self.buffers: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, dtype=torch.float32, device=self.device)
            for name, p in params if self.owners[name] == self.rank
        }
        self.microsteps = 0
        self.clipped = False
        self._hooks = []
        self._by_id = {id(p): name for name, p in params}
        self._hook_mode = False
        self._seen = set()
        self._pending = []  # (work, fp32_tensor, owner_name_or_None)
        self._use_cuda_async = (
            self.device.type == 'cuda' and self.world > 1 and dist.is_available() and dist.is_initialized()
        )
        self._stream = torch.cuda.Stream(device=self.device) if self._use_cuda_async else None
        self.preclip_norm = None

    def install_hooks(self):
        """Cast BF16 grads to FP32 and reduce-SUM to the owner as soon as each grad is final."""
        if self._hooks:
            return
        if not hasattr(torch.nn.Parameter, 'register_post_accumulate_grad_hook'):
            raise RuntimeError('torch Parameter.register_post_accumulate_grad_hook is required')

        def make_hook(name: str):
            def hook(param: torch.nn.Parameter):
                if self.clipped:
                    raise ValueError('gradient accumulator already clipped')
                grad = param.grad
                if grad is None or grad.is_sparse or grad.shape != param.shape:
                    raise ValueError('missing/dense gradient inventory: ' + name)
                owner = self.owners[name]
                with torch.no_grad():
                    if self._use_cuda_async:
                        cur = torch.cuda.current_stream(device=self.device)
                        self._stream.wait_stream(cur)
                        with torch.cuda.stream(self._stream):
                            t = grad.detach().float().contiguous()
                            work = dist.reduce(t, dst=owner, op=dist.ReduceOp.SUM, async_op=True)
                            # Owner keeps ``t`` (post-reduce SUM) to add into the shard; others drop after wait.
                            keep_name = name if owner == self.rank else None
                            self._pending.append((work, t, keep_name))
                        param.grad = None
                        while len(self._pending) > self.max_inflight:
                            self._drain_one()
                    else:
                        # world==1 or CPU/gloo: fold locally; optional sync reduce for multi-rank CPU tests.
                        t = grad.detach().float() if grad.dtype != torch.float32 else grad.detach()
                        if self.world > 1:
                            if not dist.is_available() or not dist.is_initialized():
                                raise RuntimeError('torch.distributed required for world>1')
                            dist.reduce(t, dst=owner, op=dist.ReduceOp.SUM)
                        if owner == self.rank:
                            # Match v834c FP32.add_(BF16) when world==1 (no prior float()); else add FP32 SUM.
                            if self.world == 1 and grad.dtype != torch.float32:
                                self.buffers[name].add_(grad.detach())
                            else:
                                self.buffers[name].add_(t)
                        param.grad = None
                self._seen.add(name)
                return None
            return hook

        for name, parameter in self.parameters:
            self._hooks.append(parameter.register_post_accumulate_grad_hook(make_hook(name)))
        self._hook_mode = True

    def _drain_one(self):
        work, tensor, name = self._pending.pop(0)
        work.wait()
        if name is not None:
            with torch.cuda.stream(self._stream):
                self.buffers[name].add_(tensor)
        del tensor

    def finish(self):
        """Wait for in-flight async reduces and fold owner SUMs into shards."""
        while self._pending:
            self._drain_one()
        if self._stream is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self._stream)

    def remove_hooks(self):
        self.finish()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._hook_mode = False

    def capture(self):
        """End of one micro-batch: drain async reduces and advance the microstep counter.

        With hooks installed, grads were already reduced during backward. Without hooks
        (tests / fallback), fold ``param.grad`` with the same reduce-to-owner semantics.
        """
        if self.clipped:
            raise ValueError('gradient accumulator already clipped')
        if self._hook_mode:
            self.finish()
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
                owner = self.owners[name]
                g = parameter.grad.detach()
                t = g.float() if g.dtype != torch.float32 else g
                if self.world > 1:
                    dist.reduce(t, dst=owner, op=dist.ReduceOp.SUM)
                if owner == self.rank:
                    if self.world == 1 and g.dtype != torch.float32:
                        self.buffers[name].add_(g)
                    else:
                        self.buffers[name].add_(t)
                parameter.grad = None
        self.microsteps += 1

    def local_sqnorm(self) -> torch.Tensor:
        """Sum of squared L2 norms over tensors owned by this rank (FP32)."""
        if not self.buffers:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for gradient in self.buffers.values():
            total = total + torch.linalg.vector_norm(gradient, dtype=torch.float32).square()
        return total

    def clip_shards(self, maximum: float) -> float:
        """Global-norm clip on owned GPU shards (all-reduce sum-of-squares). Same math as v834c clip."""
        if self.clipped or not self.microsteps:
            raise ValueError('one clip after nonempty accumulation')
        self.finish()
        local = self.local_sqnorm()
        if self.world > 1:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError('torch.distributed required for world>1 clip_shards')
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        total = torch.sqrt(local)
        if not torch.isfinite(total):
            raise ValueError('nonfinite accumulated gradient norm')
        scale = torch.clamp(maximum / (total + 1e-6), max=1.)
        with torch.no_grad():
            for gradient in self.buffers.values():
                gradient.mul_(scale)
        self.clipped = True
        self.preclip_norm = float(total)
        return self.preclip_norm

    def clip(self, maximum: float, *, buffers: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """Compatibility wrapper: shard clip when ``buffers`` is None; else clip a gathered dict."""
        if buffers is None:
            return self.clip_shards(maximum)
        if self.clipped or not self.microsteps:
            raise ValueError('one clip after nonempty accumulation')
        norms = [torch.linalg.vector_norm(g, dtype=torch.float32) for g in buffers.values()]
        # Gathered CPU tensors: stack on CPU.
        total = torch.linalg.vector_norm(torch.stack([n.cpu() if n.device.type != 'cpu' else n for n in norms]),
                                         dtype=torch.float32)
        if not torch.isfinite(total):
            raise ValueError('nonfinite accumulated gradient norm')
        scale = torch.clamp(maximum / (total + 1e-6), max=1.)
        with torch.no_grad():
            for gradient in buffers.values():
                gradient.mul_(scale.to(device=gradient.device))
        self.clipped = True
        self.preclip_norm = float(total)
        return self.preclip_norm

    def gather_to_cpu(self, dst: int = 0, group=None) -> Optional[Dict[str, torch.Tensor]]:
        """Materialize the full FP32 inventory on ``dst`` as pinned CPU tensors (send/recv)."""
        self.finish()
        pin = torch.cuda.is_available()
        if self.world == 1:
            full = {}
            for name, parameter in self.parameters:
                src = self.buffers[name]
                host = torch.empty(parameter.shape, dtype=torch.float32, device='cpu', pin_memory=pin)
                host.copy_(src, non_blocking=False)
                full[name] = host
            return full if self.rank == dst else None
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('torch.distributed is required for world>1 gather_to_cpu')
        full: Dict[str, torch.Tensor] = {}
        for name, parameter in self.parameters:
            owner = self.owners[name]
            if self.rank == owner:
                tensor = self.buffers[name]
            else:
                tensor = torch.empty(parameter.shape, dtype=torch.float32, device=self.device)
            if owner != dst:
                if self.rank == owner:
                    dist.send(tensor, dst=dst, group=group)
                elif self.rank == dst:
                    dist.recv(tensor, src=owner, group=group)
            if self.rank == dst:
                host = torch.empty(parameter.shape, dtype=torch.float32, device='cpu', pin_memory=pin)
                host.copy_(tensor, non_blocking=False)
                full[name] = host
            if self.rank != owner and self.rank != dst:
                del tensor
            elif self.rank == dst and owner != dst:
                del tensor
        return full if self.rank == dst else None

    # Back-compat alias used by older qual scripts / tests.
    def gather_full_buffers(self, dst: int = 0, group=None) -> Optional[Dict[str, torch.Tensor]]:
        return self.gather_to_cpu(dst=dst, group=group)

    def reduce_shards(self, group=None):
        """No-op: reduce already happened in post-accumulate hooks. Kept for call-site compatibility."""
        self.finish()

    def clear(self):
        self.remove_hooks()
        self.buffers.clear()
        self._pending.clear()
        self.microsteps = 0
        self.preclip_norm = None


def estimate_bytes(named_parameters: NamedParams, world: int = 8, max_inflight: int = 4) -> dict:
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
            # One BF16 grad in flight + up to max_inflight FP32 reduce temps.
            bf16_param_grad_peak_bytes=peak_param * 2,
            reduce_scratch_bytes=peak_param * 4 * max_inflight,
            resident_grad_related_bytes=(max(shard_numel) * 4 if shard_numel else 0)
            + peak_param * 2 + peak_param * 4 * max_inflight,
        ),
        shard_numel=shard_numel,
        largest_tensor_numel=peak_param,
        saved_vs_v834c_bytes=(total_numel * 4 + bf16_w) - (
            (max(shard_numel) * 4 if shard_numel else 0) + peak_param * 2 + peak_param * 4 * max_inflight
        ),
    )


def gb(n: int) -> float:
    return n / (1024 ** 3)
