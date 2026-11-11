"""v8.3.5 / host-v834f: ZeRO-3 execution of the v8.3.4e GRPO update on 8 low-VRAM GPUs.

host-v834e keeps on EVERY GPU a full bf16 actor, its bf16 gradient and a full fp32 gradient accumulator, plus
the activations of one micro-batch (about 112 GiB on one device). Here the same update runs with the
parameters, gradients and gradient accumulators SHARDED over the v8.3 DP world:

  sharding   torch FSDP (torch.distributed.fsdp.fully_shard). Every rank applies it to its own actor copy on the one
             default process group. Decoder layers reshard after forward (FULL_SHARD / ZeRO-3): a layer's full
             weights exist on a GPU only while that layer runs forward or backward. fully_shard works in place, so
             module and parameter names never change.
  precision  A layer's shards are the exact fp32 upcast of the bf16 actor and are all-gathered in bf16
             (MixedPrecisionPolicy.param_dtype), so every forward uses the bf16 weights bit for bit. Gradients are
             bf16 from autograd, as in v8.3.4e, then SUM-reduced in fp32 and accumulated as fp32 shards
             (reduce_dtype float32, divide factor 1): the v8.3.4e fp32 accumulator, sharded.
  head       lm_head runs once per logit chunk, a data-dependent number of calls, so hook-driven collectives would
             differ between ranks. It is all-gathered once per micro-batch and its bf16 gradient is SUM-reduced in
             fp32 by one explicit collective as soon as the head backward is complete. embed_tokens is gathered for
             its one forward only; its dense gradient is reduced the same way at the end of the micro-batch.
  lockstep   FSDP collectives need every rank to run the same sequence. Ranks run step j together; a rank with no
             micro-batch left runs a short forward/backward whose loss is multiplied by 0 (exact zero gradients).
  inputs     Gradient checkpointing keeps the input of every decoder layer until backward: 34 hidden-state tensors,
             the largest fixed item of a long micro-batch. A rank can keep them in host memory instead (bit-exact
             copies, no change to any collective). It does so when the estimate for the largest micro-batch of the
             job is near the limit, or when the measured probe step says the GPU needs it.
  optimizer  Unchanged and on rank 0: the fp32 shards are gathered into one full fp32 gradient on the CPU, then the
             v8.3.4e clip, CPUAdamW.step (fp32 head master) and publication run on rank 0's own unsharded bf16
             actor, which is kept on the CPU. Checkpoint files and keys therefore come from the same
             save_pretrained call on the same module tree as host-v834e.
  master     The fp32 head master is NOT sharded. It stays whole in rank 0's CPUAdamW, in host memory, and the
             launcher's unchanged code loads, verifies, updates and saves it: the same store entry, files and keys
             as host-v834e. The fp32 shards FSDP keeps are the bf16 weights written as fp32. They are not the
             master, and nothing here writes them to disk.
  failure    An error found at a step boundary stops every rank together through a control-group reduction. An error
             in the middle of a forward or backward cannot be re-synchronised: the rank exits and the rank-0
             watchdog stops the whole world at once instead of waiting for the 45 minute collective timeout.

Nothing here runs unless PIT_ZERO3 is on. With it off the host-v834f launcher takes the unchanged v8.3.4e path.
"""
import dataclasses
import importlib.util
import json
import math
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

HOST_VERSION = 'host-v834f'
ZERO3_ENV = 'PIT_ZERO3'
LOW_MEM_GPU_GIB = 100.0            # below this a GPU cannot hold the unsharded v8.3.4e update
LOW_VRAM_TOKEN_BUDGET = 24576      # micro-batch token budget of the sharded path (one longer rollout still gets its own bin)
LOGIT_CHUNK = 1024                 # head tokens per lm_head call (v8.3.4e: 4096): one fp32 logits tensor is 0.95 GiB instead of 3.8
TARGET_PEAK_GIB = 28.0             # per-GPU limit enforced by the measured probe step
TRAIN_MAX_CONTEXT = 36864          # = v8211_batched.TRAIN_MAX_CONTEXT: the longest single rollout a bin can hold
GIB = float(1 << 30)

# Exact geometry of the pinned 9B model (transformers Qwen3_5Config defaults = MiMo 9B), in parameters.
REFERENCE_9B = dict(layers=32, layer_max=218_407_104, layer_sum=6_919_561_728, embed=1_017_118_720,
                    head=1_017_118_720, other=4_096, frozen=453_650_672, hidden=4096, vocab=248_320)
# Activation coefficients of the peak estimate, in hidden-state tensors (tokens x hidden, compute dtype) and fp32
# logits tensors. They are hand counts of what autograd keeps for the backward of ONE recomputed block, and they
# are NOT measured. Linear-attention block (24 of 32): the two RMSNorms 6, the token mixer 24 (projections 3,
# convolution 2, repeated q/k 2, delta-rule kernel state about 6, gated norm in fp32 9, output 2), the 3x MLP 13,
# residuals 2, gradients in flight about 3.
EST_LAYER_TENSORS = 48.0
EST_HEAD_LOGITS = 5.0
OFFLOAD_INPUTS_ABOVE = 0.75        # keep the layer inputs on the host when the estimate is above this share of the limit
OFFLOADED_INPUT_TENSORS = 3        # hidden-state tensors still on the GPU when the layer inputs are on the host
EST_UNSHARDED_REFERENCE_GIB = 112.0   # reported v8.3.4e single-GPU peak of the 9B update (B300)

MAX_PROBES = 3                     # measured probe steps per job (a rank may move its layer inputs to the host and measure again)
GO, RETRY, REFUSE = 0, 1, 2        # one rank's verdict on a probe step; the world acts on the largest

WORLD = dict(dead=None, helpers=(), actor=None)


class VramPreflightError(RuntimeError):
    """The sharded update does not fit this GPU or host; refused before any gradient is produced."""


def _need(ok, why):
    if not ok:
        raise ValueError(why)


def _event(event, **fields):
    print(json.dumps(dict(component='trainer-host-v834f', event=event, **fields), default=str), flush=True)


# ---------------------------------------------------------------------------------------------- policy

def enabled():
    return os.environ.get(ZERO3_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')


def per_gpu_mem_gib():
    for key in ('PIT_PER_GPU_MEM_GIB', 'CUDA_VISIBLE_MEM_GIB'):
        raw = os.environ.get(key)
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    if torch.cuda.is_available():
        return float(torch.cuda.get_device_properties(0).total_memory) / GIB
    return None


def needs_zero3(mem_gib=None):
    if enabled():
        return True
    mem = per_gpu_mem_gib() if mem_gib is None else mem_gib
    return mem is not None and float(mem) < LOW_MEM_GPU_GIB


def token_budget(engine_budget):
    """The micro-batch token budget the sharded plan uses (applied by the launcher to v83_dp.BUDGET)."""
    return min(int(engine_budget), LOW_VRAM_TOKEN_BUDGET)


def activate(host_mod):
    """Choose the execution mode of this trainer process. Call before host.install().

    Sharded mode is carried by PIT_ZERO3: the host-v834f launcher reads it in install(), starts every helper through
    this module's helper entry and installs the sharded update. The launcher module is therefore required: without
    it nothing would start the helpers the sharded way, so activate(None) refuses instead of half-enabling the mode.
    """
    mem = per_gpu_mem_gib()
    sharded = needs_zero3(mem)
    if sharded:
        if host_mod is None or 'v835_zero3.py' not in getattr(host_mod, 'HOST_FILES', ()) \
                or not callable(getattr(host_mod, 'zero3', None)):
            raise RuntimeError('sharded mode needs the host-v834f launcher module: activate(host)')
        if host_mod.zero3() is not sys.modules[__name__]:
            raise RuntimeError('activate() must be called on the launcher\'s own module: host.zero3().activate(host)')
        os.environ[ZERO3_ENV] = '1'
    return dict(per_gpu_mem_gib=mem, zero3=sharded, host_version=HOST_VERSION, threshold_gib=LOW_MEM_GPU_GIB,
                token_budget_cap=LOW_VRAM_TOKEN_BUDGET if sharded else None,
                logit_chunk=LOGIT_CHUNK if sharded else None, target_peak_gib=TARGET_PEAK_GIB if sharded else None,
                backend=_backend() if sharded else None)


def _backend():
    return 'nccl' if torch.cuda.is_available() else 'gloo'


def helper_argv(host_dir, rank, world, port, source_root):
    """Command of one helper rank. The v83_dp.py path is an argument on purpose: the box clean-up finds trainer
    processes with `pgrep -f 'train_pit.py|v83_dp.py'`, so a sharded helper must match that pattern too."""
    host_dir = Path(host_dir)
    return [sys.executable, '-B', str(host_dir / 'v835_zero3.py'), 'helper', str(rank), str(world), str(port),
            str(host_dir / 'v83_dp.py'), str(source_root)]


# ---------------------------------------------------------------------------------------------- memory model

def inventory(model):
    """Parameter counts of the units of the sharding plan, read from the loaded model."""
    root, embed, head, layers = _units(model)
    per_layer = [sum(p.numel() for p in layer.parameters()) for layer in layers]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_embed = sum(p.numel() for p in embed.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    return dict(layers=len(per_layer), layer_max=max(per_layer), layer_sum=sum(per_layer), embed=n_embed, head=n_head,
                other=trainable - sum(per_layer) - n_embed - n_head, frozen=total - trainable,
                hidden=int(embed.weight.shape[1]), vocab=int(embed.weight.shape[0]))


def scaled_inventory(n_params):
    """The 9B geometry scaled to another parameter count (host-side planning without a loaded model)."""
    total = sum(REFERENCE_9B[k] for k in ('layer_sum', 'embed', 'head', 'other', 'frozen'))
    s = float(n_params) / float(total)
    out = dict(REFERENCE_9B)
    for k in ('layer_max', 'layer_sum', 'embed', 'head', 'other', 'frozen'):
        out[k] = int(REFERENCE_9B[k] * s)
    return out


def estimate_peak_gib(inv, *, tokens, world, logit_chunk=LOGIT_CHUNK, compute_bytes=2, offload_inputs=False):
    """Per-GPU peak of one sharded micro-batch of `tokens` tokens.

    `certain_gib` adds only tensors the design keeps alive by construction (exact sizes). `estimate_gib` adds the
    activation transients with the EST_* coefficients, which are counted from the code and not measured. The
    measured gate is the probe step of the update (see _run); this function refuses only what cannot fit.
    `offload_inputs`: the checkpointed layer inputs are in host memory (`offload_host_gib` per rank) instead.
    """
    world = max(int(world), 1)
    hid = tokens * inv['hidden'] * compute_bytes / GIB               # one hidden-state tensor
    kept = (inv['layers'] + 2) * hid                                 # checkpointing keeps every layer input, the final states and the selected states
    saved = min(kept, OFFLOADED_INPUT_TENSORS * hid) if offload_inputs else kept
    unit = inv['layer_max'] * compute_bytes / GIB                    # one layer, compute dtype
    head_w = inv['head'] * compute_bytes / GIB
    embed_w = inv['embed'] * compute_bytes / GIB
    shards = (head_w + embed_w) / world                              # resident: the two unoffloaded shard sets
    logits = min(logit_chunk, tokens) * inv['vocab'] * 4 / GIB       # one fp32 logits tensor
    total = sum(inv[k] for k in ('layer_sum', 'embed', 'head', 'other', 'frozen')) * compute_bytes / GIB
    phases = {
        # start-up: every rank loads the whole actor once, then shards it unit by unit (one layer upcast to fp32)
        'load': (total + 2 * unit, 0.0),
        # one head chunk: lm_head weight + its accumulating gradient + bf16 logits, fp32 logits and their fp32 gradient
        'head': (shards + saved + 2 * head_w + 2.5 * logits, (EST_HEAD_LOGITS - 2.5) * logits),
        # explicit lm_head reduction: weight + bf16 gradient + fp32 copy + fp32 shard
        'head-reduce': (shards + saved + 2 * head_w + 2 * head_w + 2 * head_w / world, 0.0),
        # one layer backward: this layer and the prefetched next one unsharded, bf16 gradient, the previous and
        # the current fp32 reduce-scatter input, one fp32 output shard
        'layer': (shards + saved + 2 * unit + unit + 4 * unit + 2 * unit / world, EST_LAYER_TENSORS * hid),
        # explicit embedding reduction after backward: dense bf16 gradient + fp32 copy + fp32 shard
        'embed-reduce': (shards + embed_w + 2 * embed_w + 2 * embed_w / world, 0.0),
    }
    certain = max(v[0] for v in phases.values())
    estimate = max(v[0] + v[1] for v in phases.values())
    return dict(certain_gib=certain, estimate_gib=estimate, peak_gib=estimate, tokens=int(tokens), world=world,
                logit_chunk=int(logit_chunk), offload_inputs=bool(offload_inputs),
                offload_host_gib=(kept - saved) if offload_inputs else 0.0,
                phases={k: dict(certain_gib=round(v[0], 3), estimate_gib=round(v[0] + v[1], 3)) for k, v in phases.items()},
                coefficients=dict(layer_tensors=EST_LAYER_TENSORS, head_logits=EST_HEAD_LOGITS, measured=False))


def host_ram_gib(inv, *, compute_bytes=2):
    """Host memory (CPU) of the sharded update, all ranks together, in GiB. Counted from the design, not measured.

    rank0_gib   the trainer process before a job: its unsharded actor and the whole fp32 master (as in v8.3.4e)
    shards_gib  the fp32 parameter shards of the decoder layers (CPU offload)
    step_gib    what a job adds: the fp32 gradient (in shards while it accumulates, then gathered on rank 0; a
                shard is released when it is sent) and the fp32/fp64 temporaries of the optimizer step
    """
    trainable = inv['layer_sum'] + inv['embed'] + inv['head'] + inv['other']
    largest = max(inv['embed'], inv['head'], inv['layer_max'])
    step = (trainable * 4 + largest * 20) / GIB
    rank0 = ((trainable + inv['frozen']) * compute_bytes + trainable * 4) / GIB
    shards = (inv['layer_sum'] + inv['other']) * 4 / GIB
    return dict(step_gib=step, rank0_gib=rank0, shards_gib=shards, total_gib=rank0 + shards + step)


def limit_gib(per_gpu_gib):
    return min(TARGET_PEAK_GIB, 0.9 * float(per_gpu_gib))


def preflight_or_raise(*, per_gpu_gib, world=8, n_params=None, inv=None, token_budget=None, tokens=None,
                       sharded=True, logit_chunk=LOGIT_CHUNK):
    """Refuse a configuration that cannot fit. The sharded branch models what the sharded update keeps on a GPU
    (not total / world); the unsharded branch is the reported v8.3.4e peak scaled by parameter count."""
    if per_gpu_gib is None:
        raise VramPreflightError('per-GPU VRAM unknown; refuse to start the sharded update')
    inv = inv if inv is not None else scaled_inventory(9.4e9 if n_params is None else n_params)
    if not sharded:
        total = sum(inv[k] for k in ('layer_sum', 'embed', 'head', 'other', 'frozen'))
        peak = EST_UNSHARDED_REFERENCE_GIB * total / sum(REFERENCE_9B[k] for k in ('layer_sum', 'embed', 'head', 'other', 'frozen'))
        out = dict(peak_gib=peak, certain_gib=peak, estimate_gib=peak, sharded=False, per_gpu_gib=float(per_gpu_gib), world=int(world))
        if peak > float(per_gpu_gib):
            raise VramPreflightError('unsharded update needs about %.1f GiB per GPU and exceeds VRAM %.1f GiB' % (peak, per_gpu_gib))
        return out
    if tokens is None:      # worst micro-batch: the budget, or one rollout of the maximum training context
        tokens = max(LOW_VRAM_TOKEN_BUDGET if token_budget is None else min(int(token_budget), LOW_VRAM_TOKEN_BUDGET), TRAIN_MAX_CONTEXT)
    limit = limit_gib(per_gpu_gib)
    out = estimate_peak_gib(inv, tokens=tokens, world=world, logit_chunk=logit_chunk)
    if out['estimate_gib'] > OFFLOAD_INPUTS_ABOVE * limit:           # near the limit: plan with the layer inputs on the host
        out = estimate_peak_gib(inv, tokens=tokens, world=world, logit_chunk=logit_chunk, offload_inputs=True)
    out.update(sharded=True, per_gpu_gib=float(per_gpu_gib), limit_gib=limit)
    if out['certain_gib'] > limit:
        raise VramPreflightError('sharded update keeps at least %.1f GiB on a GPU at %d tokens and exceeds VRAM limit %.1f GiB (card %.1f GiB, world %d)'
                                 % (out['certain_gib'], tokens, limit, per_gpu_gib, world))
    return out


def _card_gib():
    """The card this rank really has. The PIT_PER_GPU_MEM_GIB value only stands in where there is no GPU."""
    if torch.cuda.is_available():
        return float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory) / GIB
    return per_gpu_mem_gib()


def preflight_for_runtime(runtime, *, world=8, token_budget=None):
    return preflight_or_raise(per_gpu_gib=_card_gib(), world=world, inv=inventory(runtime.model),
                              token_budget=token_budget)


def _mem_available_gib():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return float(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------------------------- sharded actor

def _units(model):
    """(root, embed, head, layers) of the pinned HF Qwen3.5 layout; anything else is refused."""
    root = model.model
    lm = root.language_model
    embed, head, layers = lm.embed_tokens, model.lm_head, list(lm.layers)
    _need(len(layers) > 0 and model.get_input_embeddings() is embed, 'Qwen3.5 text decoder layout required')
    _need(all(a is not b for a in head.parameters() for b in embed.parameters()),
          'tied lm_head / embedding weights are not supported by the sharded plan')
    return root, embed, head, layers


def _dim0_rows(rows, world, rank):
    """(chunk, start, length) of rank's dim-0 shard: torch.chunk semantics, as FSDP and DTensor Shard(0) use."""
    chunk = -(-rows // world)
    start = min(rank * chunk, rows)
    return chunk, start, max(0, min(chunk, rows - start))


def _wire(tensor):
    """The gloo control group does not carry bf16: any non-fp32 tensor travels as its raw bytes."""
    return tensor if tensor.dtype == torch.float32 else tensor.view(torch.uint8)


class _SavedInputsOnHost(torch.autograd.graph.saved_tensors_hooks):
    """Tensors autograd saves while this is active go to host memory and come back when backward asks for them.

    It is active around one checkpointed layer call. Inside that call the checkpoint replaces every saved tensor
    of the block by a placeholder, so the only tensor that reaches these hooks is what the checkpoint itself
    keeps for the recomputation: the layer input. The copies are bit-exact and no collective changes.
    """

    def __init__(self):
        super().__init__(self._pack, self._unpack)

    @staticmethod
    def _pack(tensor):
        return tensor.device, tensor.to('cpu', copy=True)

    @staticmethod
    def _unpack(packed):
        device, tensor = packed
        return tensor.to(device)


class Actor:
    """One rank's FSDP-sharded copy of the actor and the collectives of one micro-batch step."""

    def __init__(self, runtime, world, rank):
        self.runtime, self.model, self.world, self.rank = runtime, runtime.model, world, rank
        self.root, self.embed, self.head, self.layers = _units(self.model)
        self.device = runtime.device
        self.cuda = self.device.type == 'cuda'
        self.compute_dtype = self.embed.weight.dtype
        self.inv = inventory(self.model)
        # canonical trainable inventory = the names and order CPUAdamW sees on the unsharded actor
        self.names = [(n, tuple(p.shape)) for n, p in self.model.named_parameters() if p.requires_grad]
        _need(all(len(shape) >= 1 for _, shape in self.names), 'FSDP shards dim 0: 0-dim parameters are not supported')
        self.accumulate = True
        self.acc = {}                 # explicit groups: fp32 CPU shard accumulators
        self.probe_max = 0.0
        self.explicit = {}
        self.offload_inputs = False   # keep the checkpointed layer inputs in host memory (this rank's own choice)
        self._shard()
        self._wrap_layer_inputs()

    def _shard(self):
        """Apply FSDP to every unit. All ranks run this together: it issues collectives."""
        mesh = init_device_mesh(self.device.type, (self.world,))
        offload = CPUOffloadPolicy(pin_memory=False) if self.cuda else OffloadPolicy()
        native = MixedPrecisionPolicy(param_dtype=self.compute_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False)
        frozen = {p for p in self.root.parameters() if not p.requires_grad}
        for child in list(self.root.children()) + [m for m in self.model.children() if m is not self.root and m is not self.head]:
            # Only the flags are kept. A list of the child's Parameters would still be bound when the units below
            # are sharded: fully_shard replaces a Parameter by this rank's shard, and the full-size tensor of the
            # original (after the fp32 upcast: twice the bf16 size) is freed only when nothing else refers to it.
            trainable = [p.requires_grad for p in child.parameters()]
            if trainable and not any(trainable):
                child.to('cpu')       # frozen and outside the text forward (vision tower): keep it off the GPU

        def fp32(module):             # exact upcast: the shards and their gradients are fp32, the forward stays bf16
            for p in module.parameters():       # p is local to this call: it must not outlive it (see above)
                if p.requires_grad and not isinstance(p, DTensor) and p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32)

        def sum_reduce(module):
            module.set_gradient_divide_factor(1.0)             # SUM over ranks, as v8.3.4e reduces its accumulators
            module.set_force_sum_reduction_for_comms(True)
            module.set_reduce_scatter_unused_params(True, recurse=False)

        for layer in self.layers:
            fp32(layer)
            fully_shard(layer, mesh=mesh, reshard_after_forward=True, mp_policy=native, offload_policy=offload)
            sum_reduce(layer)
        # embedding: gathered for its one forward, not needed in backward; its gradient is reduced by end_step()
        fully_shard(self.embed, mesh=mesh, reshard_after_forward=True)
        self.embed.set_requires_gradient_sync(False, recurse=False)
        self.embed.set_unshard_in_backward(False)
        # lm_head: gathered by begin_step(), kept for every logit chunk, reduced and resharded by _head_done()
        fully_shard(self.head, mesh=mesh, reshard_after_forward=False)
        self.head.set_requires_gradient_sync(False, recurse=False)
        self.head.set_reshard_after_backward(False, recurse=False)
        # root (model.model): the remaining trainable parameters (final norm); frozen ones stay unmanaged
        fp32(self.root)               # the units sharded above are DTensors by now and are skipped
        fully_shard(self.root, mesh=mesh, mp_policy=native, offload_policy=offload, ignored_params=frozen)
        sum_reduce(self.root)
        for unit in self.layers + [self.embed, self.head, self.root]:
            _need(isinstance(unit, FSDPModule), 'every unit must be FSDP-managed')
        # The unsharded Parameter of an FSDP parameter is one persistent object. Hold it for the two explicit
        # units so their autograd gradients can be read after the unit has been resharded.
        for kind, unit in (('embed', self.embed), ('head', self.head)):
            unit.unshard()
            index = {id(p): n for n, p in self.model.named_parameters()}
            self.explicit[kind] = [(index[id(p)], p) for p in unit.parameters() if p.requires_grad]
            unit.reshard()
        self.root.register_forward_hook(self._root_output)
        explicit = {n for rows in self.explicit.values() for n, _ in rows}
        self.native = [n for n, _ in self.names if n not in explicit]
        params = dict(self.model.named_parameters())
        _need([n for n, _ in self.names] == [n for n, p in params.items() if p.requires_grad]
              and all(isinstance(params[n], DTensor) for n, _ in self.names),
              'sharding must keep the exact trainable names, all as sharded parameters')
        for n, shape in self.names:
            _need(tuple(params[n].shape) == shape and tuple(params[n].to_local().shape[1:]) == shape[1:]
                  and params[n].to_local().shape[0] == _dim0_rows(shape[0], self.world, self.rank)[2],
                  'unexpected shard geometry: ' + n)

    def _wrap_layer_inputs(self):
        """Put the host-memory switch around each layer's checkpointed forward (the runtime's instance wrapper)."""
        def switched(inner):
            def forward(*args, **kwargs):
                if not self.offload_inputs:
                    return inner(*args, **kwargs)
                with _SavedInputsOnHost():
                    return inner(*args, **kwargs)
            return forward
        for layer in self.layers:
            _need('forward' in vars(layer), 'every decoder layer must run under the runtime gradient checkpoint')
            layer.forward = switched(layer.forward)

    # -- weights

    def sync(self, ctrl, canonical=None):
        """Write rank 0's actor into every rank's shards, bit for bit, one tensor at a time (control group)."""
        params = dict(self.model.named_parameters())
        for name, shape in self.names:
            local = params[name].to_local()
            chunk = _dim0_rows(shape[0], self.world, self.rank)[0]
            wire = torch.empty((chunk,) + shape[1:], dtype=self.compute_dtype)
            pieces = None
            if self.rank == 0:
                src = canonical[name].detach()
                _need(tuple(src.shape) == shape and src.dtype == self.compute_dtype and src.device.type == 'cpu',
                      'canonical actor tensor: ' + name)
                pad = chunk * self.world - shape[0]
                if pad:
                    src = F.pad(src, [0, 0] * (src.dim() - 1) + [0, pad])
                pieces = [_wire(piece.contiguous()) for piece in src.split(chunk, dim=0)]
            dist.scatter(_wire(wire), scatter_list=pieces, src=0, group=ctrl)
            with torch.no_grad():
                local.copy_(wire[:local.shape[0]].to(device=local.device, dtype=local.dtype))

    # -- one micro-batch step

    def begin_step(self):
        self.head.unshard()

    def _root_output(self, module, args, output):
        hidden = getattr(output, 'last_hidden_state', None)
        if torch.is_grad_enabled() and hidden is not None and hidden.requires_grad:
            hidden.register_hook(self._head_done)
        return None

    def _head_done(self, grad):
        """Backward reached the decoder output: every head chunk is done. Reduce lm_head and free its weights
        before the layers run backward. Same graph position on every rank."""
        self._reduce('head')
        self.head.reshard()
        return None

    def end_step(self):
        self._reduce('embed')
        for kind in ('embed', 'head'):
            for name, ref in self.explicit[kind]:
                _need(ref.grad is None, 'unreduced gradient after the step: ' + name)

    def _reduce(self, kind):
        """SUM-reduce the bf16 gradients of one explicit unit in fp32 and keep this rank's shard."""
        for name, ref in self.explicit[kind]:
            grad, ref.grad = ref.grad, None
            if grad is None:          # unused in this graph: contribute zeros so every rank sends the same payload
                grad = torch.zeros(ref.shape, dtype=self.compute_dtype, device=self.device)
            grad = grad.detach().to(torch.float32)
            chunk, _, length = _dim0_rows(grad.shape[0], self.world, self.rank)
            pad = chunk * self.world - grad.shape[0]
            if pad:
                grad = F.pad(grad, [0, 0] * (grad.dim() - 1) + [0, pad])
            # the collective FSDP itself uses: rank r keeps the r-th equal slice of the flat SUM
            out = torch.empty(grad.numel() // self.world, dtype=torch.float32, device=grad.device)
            dist.reduce_scatter_single(output=out, input=grad.reshape(-1), op=dist.ReduceOp.SUM)
            shard = out.view((chunk,) + tuple(grad.shape[1:]))[:length].to('cpu')
            if self.accumulate:
                if name in self.acc:
                    self.acc[name].add_(shard)
                else:
                    self.acc[name] = shard.clone()
            elif shard.numel():
                self.probe_max = max(self.probe_max, float(shard.abs().max()))

    def null_step(self, B, items):
        """Same module calls and collectives as a real micro-batch, contributing exact zeros."""
        with torch.enable_grad():
            outs = B.batched_log_probs(self.runtime, items, gradients=True, engine='packed', logit_chunk=LOGIT_CHUNK,
                                       support=B.SUPPORT_RENORM)
        terms = [t.sum() for out in outs for t in (out if isinstance(out, tuple) else (out,))
                 if t.is_floating_point() and t.requires_grad]
        total = torch.stack(terms).sum()
        if not torch.isfinite(total):
            raise ValueError('non-finite forward in a zero-weight step')
        (total * 0.0).backward()
        self.end_step()

    # -- gradients

    def shard_grad(self, name, params=None):
        """This rank's fp32 CPU shard of the accumulated gradient of one parameter (zeros when nothing arrived)."""
        if name in self.acc:
            return self.acc[name]
        param = (params or dict(self.model.named_parameters()))[name]
        grad = param.grad
        if grad is None:
            return torch.zeros(tuple(param.to_local().shape), dtype=torch.float32)
        local = grad.to_local()
        _need(local.dtype == torch.float32, 'fp32 gradient shard required: ' + name)
        return local.to('cpu')

    def gather(self, ctrl, buffers=None):
        """Send every fp32 gradient shard to rank 0 (control group). Rank 0 fills `buffers` (full fp32, CPU).
        A shard is released as soon as it is copied out, so host memory never holds the gradient twice."""
        params = dict(self.model.named_parameters())
        for name, shape in self.names:
            chunk, _, length = _dim0_rows(shape[0], self.world, self.rank)
            shard = self.shard_grad(name, params)
            _need(tuple(shard.shape) == (length,) + shape[1:], 'gradient shard geometry: ' + name)
            wire = torch.zeros((chunk,) + shape[1:], dtype=torch.float32)
            wire[:length].copy_(shard)
            del shard
            self.acc.pop(name, None)
            params[name].grad = None
            pieces = [torch.empty_like(wire) for _ in range(self.world)] if self.rank == 0 else None
            dist.gather(wire, gather_list=pieces, dst=0, group=ctrl)
            if self.rank == 0:
                full = buffers[name]
                _need(tuple(full.shape) == shape and full.dtype == torch.float32, 'gradient buffer: ' + name)
                for r, piece in enumerate(pieces):
                    _, start, n = _dim0_rows(shape[0], self.world, r)
                    full[start:start + n].copy_(piece[:n])

    def max_abs_grad(self):
        params = dict(self.model.named_parameters())
        shards = [self.shard_grad(n, params) for n in self.native]
        return max([self.probe_max] + [float(s.abs().max()) for s in shards if s.numel()])

    def clear(self):
        for _, param in self.model.named_parameters():
            param.grad = None
        for rows in self.explicit.values():
            for _, ref in rows:
                ref.grad = None
        self.acc.clear()
        self.probe_max = 0.0

    def shard_bytes(self):
        return {n: int(p.to_local().numel() * p.to_local().element_size())
                for n, p in self.model.named_parameters() if p.requires_grad}


class _Sink:
    """The accumulator interface v8211_batched._gradient_pass uses: capture() after each backward."""

    def __init__(self, actor):
        self.actor, self.microsteps, self.clipped, self.buffers = actor, 0, False, {}

    def capture(self):
        self.actor.end_step()
        self.microsteps += 1


# ---------------------------------------------------------------------------------------------- DP world

def init(dp, world, port, helpers):
    """Rank 0 joins the DP world. CUDA: the unchanged v8.3 groups (NCCL data + gloo control). No CUDA: gloo for
    both, which subnet.mimo_runtime only accepts for CPU fixture profiles."""
    WORLD.update(dead=None, helpers=tuple(helpers), actor=None)
    threading.Thread(target=_watch_helpers, name='v835-helper-watchdog', daemon=True).start()
    if torch.cuda.is_available():
        dp.init(world, port)
    else:
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group('gloo', rank=0, world_size=world, timeout=timedelta(minutes=45))
        dp.STATE['ctrl'] = dist.new_group(backend='gloo', timeout=timedelta(days=30))
        dp.STATE.update(world=world, initialized=True)


def _watch_helpers():
    """A helper that exits with an error can no longer take part in collectives: stop the trainer now. Blocked
    NCCL collectives cannot be interrupted, so this is the only way not to wait for their 45 minute timeout."""
    while True:
        time.sleep(1.0)
        for index, helper in enumerate(WORLD['helpers']):
            code = helper.poll()
            if code not in (None, 0):
                _event('zero3-world-abort', reason='helper exited', rank=index + 1, returncode=code)
                _kill_helpers()
                os._exit(70)


def _kill_helpers():
    for helper in WORLD['helpers']:
        try:
            helper.kill()
        except Exception:
            pass


def _agree(ctrl, world, ok, error=None):
    """Every rank reports its state at a step boundary; all ranks get the same list of errors ([] = go on)."""
    flag = torch.tensor([1.0 if ok else 0.0])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ctrl)
    if flag.item() >= 1:
        return []
    errors = [None] * world
    dist.all_gather_object(errors, None if ok else str(error), group=ctrl)
    return [e for e in errors if e] or ['unknown']


def _worst(ctrl, verdict):
    """The largest verdict over all ranks (control group)."""
    flag = torch.tensor([float(verdict)])
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=ctrl)
    return int(flag.item())


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                  # cached blocks of earlier work would count as reserved memory
        torch.cuda.reset_peak_memory_stats()


def _measured_peak():
    """This rank's peak GPU memory since the last reset, in GiB (None when there is no GPU to measure)."""
    if not torch.cuda.is_available():
        return None
    return dict(allocated_gib=torch.cuda.max_memory_allocated() / GIB, reserved_gib=torch.cuda.max_memory_reserved() / GIB)


def _probe(B, actor, items, rank):
    """One zero-weight step on the largest micro-batch of the job, on every rank at once.
    -> (peak GPU memory or None, largest gradient value the step left behind)."""
    try:
        actor.accumulate = False
        _reset_peak()
        actor.null_step(B, items)
        stray = actor.max_abs_grad()
        peak = _measured_peak()
    except BaseException as x:
        _fatal(rank, 'probe', x)
    finally:
        actor.accumulate = True
    actor.clear()
    return peak, stray


def _load_actor(B, FK, dp, job, world, rank):
    # subnet.* resolves only after the launcher / helper entry put the admitted source root on sys.path
    from subnet.mimo_runtime import MimoRuntime
    cuda = torch.cuda.is_available()
    runtime = MimoRuntime.load(job['actor_dir'], job['actor_desc'], job['spec'], device='cuda' if cuda else 'cpu',
                               trainable=True, allow_fixture=not cuda)
    if cuda:
        dp._install_fast(B, FK, runtime, 0)       # the v8.3 training kernels; every layer stays checkpointed
    actor = Actor(runtime, world, rank)
    dist.barrier()
    if cuda:
        torch.cuda.empty_cache()                  # drop the blocks of the unsharded load: the probe measures reserved memory
    return actor


def _lockstep(assign, rows):
    """Order each rank's micro-batches by size so that step j pairs micro-batches of similar cost."""
    tokens = lambda batch: sum(len(rows[i]['group'].prompt) + len(rows[i]['rollout'].tokens) for i in batch)
    return [sorted(batches, key=lambda b: (-tokens(b), b)) for batches in assign]


def _items(rows, batch):
    return [(rows[i]['group'].prompt, rows[i]['rollout'].tokens) for i in batch]


def _run(B, dp, job, rows, rank, *, config, agg_config, loss_scale, active_rollouts, guard, record, canonical=None,
         buffers=None):
    """One rank's part of a sharded job. Returns (errors, details, losses, info); errors == [] means the fp32
    gradient is complete on rank 0. Every collective below is issued by all ranks in the same order."""
    world, ctrl, actor = dp.STATE['world'], dp.STATE['ctrl'], WORLD['actor']
    details, losses = {}, {}
    actor.clear()
    actor.sync(ctrl, canonical)
    versions = B._versions(actor.model)
    # preflight on the real plan: the largest micro-batch of the job, this rank's card and host
    tokens = sum(len(p) + len(o) for p, o in _items(rows, job['probe']))
    card, free, ram = _card_gib(), _mem_available_gib(), host_ram_gib(actor.inv)
    limit = None if card is None else limit_gib(card)
    # host memory the layer inputs of this micro-batch need when every rank keeps them on the host
    on_host = world * estimate_peak_gib(actor.inv, tokens=tokens, world=world, offload_inputs=True)['offload_host_gib']
    room = free is None or free >= ram['step_gib'] + on_host
    error, fit = None, None
    try:
        if actor.cuda:
            fit = preflight_or_raise(per_gpu_gib=card, world=world, inv=actor.inv, tokens=tokens)
            if rank == 0 and free is not None and free < ram['step_gib']:
                raise VramPreflightError('host memory: %.1f GiB available, the rank-0 gradient and optimizer step need %.1f GiB'
                                         % (free, ram['step_gib']))
            if rank == 0:
                _event('zero3-preflight', tokens=tokens, certain_gib=round(fit['certain_gib'], 2),
                       estimate_gib=round(fit['estimate_gib'], 2), limit_gib=fit['limit_gib'],
                       offload_inputs=bool(fit['offload_inputs'] and room), offload_host_gib=round(on_host, 1),
                       host_ram_gib={k: round(v, 1) for k, v in ram.items()}, mem_available_gib=free)
    except VramPreflightError as x:
        error = 'rank %d: %s' % (rank, x)
    errors = _agree(ctrl, world, error is None, error)
    if errors:
        return errors, details, losses, dict(refused='preflight')
    # Measured probe. A rank whose peak is over the limit moves its layer inputs to the host; then every rank runs
    # the probe again, because the probe step issues collectives. What still does not fit is refused.
    actor.offload_inputs = bool(fit is not None and fit['offload_inputs'] and room)
    probes = []
    while True:
        peak, stray = _probe(B, actor, _items(rows, job['probe']), rank)
        probes.append(dict(peak or {}, offload_inputs=actor.offload_inputs))
        error, verdict = None, GO
        if stray != 0.0:
            error, verdict = 'rank %d: zero-weight probe left a gradient of %r' % (rank, stray), REFUSE
        elif peak is not None and limit is not None and peak['reserved_gib'] > limit:
            error = 'rank %d: probe peak %.2f GiB reserved (%.2f allocated) exceeds VRAM limit %.1f GiB at %d tokens (layer inputs on the %s)' % (
                rank, peak['reserved_gib'], peak['allocated_gib'], limit, tokens, 'host' if actor.offload_inputs else 'GPU')
            verdict = RETRY if not actor.offload_inputs and room and len(probes) < MAX_PROBES else REFUSE
            if verdict == REFUSE and not actor.offload_inputs:
                error += '; host memory: %.1f GiB available, %.1f GiB needed to keep them on the host' % (
                    -1.0 if free is None else free, ram['step_gib'] + on_host)
        worst = _worst(ctrl, verdict)
        if worst == GO:
            break
        if worst == REFUSE:
            errors = [None] * world
            dist.all_gather_object(errors, error, group=ctrl)
            return [e for e in errors if e] or ['unknown'], details, losses, dict(refused='probe', probes=probes)
        if verdict == RETRY:
            actor.offload_inputs = True
    if B._versions(actor.model) != versions:
        _fatal(rank, 'probe', ValueError('sharded actor changed during the probe'))
    sink = _Sink(actor)
    if rank == 0:       # the inactive rows are reported once, by rank 0, as in v8.3.4e
        B._gradient_pass(actor.runtime, rows, [], config, agg_config, loss_scale, active_rollouts, sink, versions,
                         details, losses, engine='packed', logit_chunk=LOGIT_CHUNK, record=record)
    live = lambda row: None if row.get('stage') == 'masked-unresolved' else record(row)
    mine, steps, null = job['assign'][rank], job['steps'], _items(rows, job['null'])
    for j in range(steps):
        error = None
        try:
            guard()
            if B._versions(actor.model) != versions:
                raise ValueError('sharded actor mutated before the single optimizer boundary')
        except Exception as x:
            error = 'rank %d: %s: %s' % (rank, type(x).__name__, x)
        errors = _agree(ctrl, world, error is None, error)
        if errors:
            actor.clear()
            return errors, details, losses, dict(refused='step', step=j)
        try:
            if j < len(mine):
                B._gradient_pass(actor.runtime, rows, [mine[j]], config, agg_config, loss_scale, active_rollouts, sink,
                                 versions, details, losses, engine='packed', logit_chunk=LOGIT_CHUNK, record=live)
            else:
                actor.null_step(B, null)
        except BaseException as x:
            _fatal(rank, 'step %d' % j, x)
    actor.gather(ctrl, buffers)
    info = dict(probes=probes, offload_inputs=actor.offload_inputs, peak=_measured_peak(), null_steps=steps - len(mine),
                shard_bytes=sum(actor.shard_bytes().values()))
    actor.clear()
    return [], details, losses, info


def _fatal(rank, where, error):
    """A failure inside a forward or backward leaves this rank out of step with the others for good."""
    _event('zero3-rank-failed', rank=rank, where=where, error='%s: %s' % (type(error).__name__, error))
    if rank != 0:
        os._exit(71)
    WORLD['dead'] = 'rank 0 failed in %s: %s: %s' % (where, type(error).__name__, error)
    _kill_helpers()
    raise error


def make_train_update(dp, B, FK):
    """-> train_update_batched-compatible sharded update (the contract of v83_dp.make_train_update)."""
    original = B.train_update_batched
    stock = B.batched_log_probs

    def log_probs(runtime, items, *, gradients, engine, logit_chunk=2048, support=None):
        actor = WORLD['actor']
        if actor is None or runtime is not actor.runtime:
            raise ValueError('the sharded forward runs on the sharded actor only')
        runtime.guard()
        if gradients and not runtime.trainable:
            raise ValueError('inference runtime cannot train')
        actor.begin_step()
        if actor.cuda:
            return FK.packed_log_probs(runtime, items, gradients=gradients, logit_chunk=LOGIT_CHUNK, support=support)
        # CPU fixture profiles only: the stock transformers forward (no CUDA kernels), same loss code
        return stock(runtime, items, gradients=gradients, engine='padded', logit_chunk=LOGIT_CHUNK, support=support)

    def train_update_dp(runtime, optimizer, groups, binding, *, config, guard=lambda: None, record=lambda row: None,
                        reference_runtime=None, engine='packed', token_budget=36864, max_rows=64, logit_chunk=2048,
                        timings=None):
        groups = tuple(groups)
        if not groups:
            return original(runtime, optimizer, groups, binding, config=config, guard=guard, record=record,
                            reference_runtime=reference_runtime, engine='padded', token_budget=token_budget,
                            max_rows=max_rows, logit_chunk=logit_chunk, timings=timings)
        return _update(dp, B, FK, runtime, optimizer, groups, binding, config, guard, record, reference_runtime, timings)

    B.batched_log_probs = log_probs
    train_update_dp._original = original
    return train_update_dp


def _update(dp, B, FK, runtime, optimizer, groups, binding, config, guard, record, reference_runtime, timings):
    # subnet.* resolves only after the launcher put the admitted source root on sys.path
    from subnet.mimo_grpo.data import digest
    from subnet.mimo_grpo.optimizer import FP32Accumulator
    if WORLD['dead']:
        raise RuntimeError('the sharded DP world was stopped (%s); restart the trainer' % WORLD['dead'])
    dp.PRE.validate(runtime, optimizer, groups, binding, config, reference_runtime)
    if reference_runtime is not None or config.kl_coefficient > 0:
        raise ValueError('the sharded update has no KL reference pass (v8.3.4: KL is off)')
    agg_config, loss_scale, aggregation = B.loss_aggregation(config)
    timings = {} if timings is None else timings
    started = time.monotonic(); guard()
    world, ctrl = dp.STATE['world'], dp.STATE['ctrl']
    base_report = dp.PRE.base_report(B, runtime, optimizer, groups, binding, config)
    actor = WORLD['actor']
    if actor is not None and actor.runtime.checkpoint_id != runtime._checkpoint_guard.descriptor['id']:
        raise RuntimeError('the sharded DP world is bound to head %s; restart the trainer for another head'
                           % actor.runtime.checkpoint_id)
    if runtime.device.type != 'cpu':          # rank 0's own actor is the optimizer / publication copy: off the GPU
        runtime.model.to('cpu')
        torch.cuda.empty_cache()
    runtime.model.eval(); versions = B._versions(runtime.model)
    if B.DROP_UNRESOLVED:
        rows, unresolved_filter = B.layout_drop_unresolved(groups, config)
    else:
        rows, unresolved_filter = B._layout(groups, config), dict(enabled=False)
    record(dict(stage='unresolved-filter', **unresolved_filter))
    active = [i for i, row in enumerate(rows) if row['active']]
    assign, nbins, loads = dp._plan(rows, active, world)
    assign = _lockstep(assign, rows)
    length = lambda i: len(rows[i]['group'].prompt) + len(rows[i]['rollout'].tokens)
    bins = [batch for batches in assign for batch in batches]
    probe = max(bins, key=lambda b: (sum(length(i) for i in b), b))
    job = dict(op='job', groups=groups, config=config, drop=B.DROP_UNRESOLVED, mode=B.LOSS_AGG_MODE, scale=B.LOSS_SCALE,
               support=B.SUPPORT_RENORM, knobs=getattr(B, 'KNOBS', None),
               actor_dir=str(runtime._checkpoint_guard.directory), actor_desc=runtime._checkpoint_guard.descriptor,
               spec=runtime.spec.to_dict(), assign=assign, steps=max(len(a) for a in assign), probe=probe,
               null=[min(active, key=lambda i: (length(i), i))])
    t = time.monotonic()
    dp._bcast(job)
    try:
        if WORLD['actor'] is None:
            WORLD['actor'] = _load_actor(B, FK, dp, job, world, 0)
        canonical = dict(runtime.model.named_parameters())
        accumulator = FP32Accumulator(runtime.model.named_parameters())      # full fp32, on the CPU with the actor
        _need([n for n, _ in accumulator.parameters] == [n for n, _ in WORLD['actor'].names],
              'sharded actor and optimizer must name the same trainable tensors')
    except BaseException as x:
        _fatal(0, 'sharding', x)
    timings['dp_sync_seconds'] = time.monotonic() - t
    prep = time.monotonic()
    errors, details, losses, info = _run(B, dp, job, rows, 0, config=config, agg_config=agg_config, loss_scale=loss_scale,
                                         active_rollouts=len(active), guard=guard, record=record, canonical=canonical,
                                         buffers=accumulator.buffers)
    if errors:
        accumulator.buffers.clear()
        if info.get('refused') in ('preflight', 'probe'):
            raise VramPreflightError('sharded update refused before any gradient: ' + '; '.join(errors))
        raise RuntimeError('v8.3.5 ZeRO-3 update failed before the optimizer step: ' + '; '.join(errors))
    gathered = [None] * world
    dist.all_gather_object(gathered, (details, losses, {i: (rows[i]['keep'], rows[i]['derail_index']) for i in details}, info),
                           group=ctrl)
    all_d, all_l, ranks = {}, {}, []
    for d, l, keep, rank_info in gathered:
        all_d.update(d); all_l.update(l); ranks.append(rank_info)
        for i, (k, di) in keep.items():
            rows[i]['keep'], rows[i]['derail_index'] = k, di
    for i in sorted(all_d):     # trajectory-loss events for helper-owned rollouts (rank-0 ones were recorded live)
        if i not in details:
            record(dict(stage='trajectory-loss', **all_d[i]))
    timings['gradient_seconds'] = time.monotonic() - prep
    accumulator.microsteps = nbins
    norm = accumulator.clip(config.max_gradient_norm)
    guard()
    if B._versions(runtime.model) != versions:
        raise ValueError('actor changed before optimizer step')
    step_started = time.monotonic()
    update = optimizer.step(accumulator.buffers)
    if optimizer.global_step != binding.optimizer_step + 1:
        raise ValueError('exactly one optimizer step required')
    accumulator.buffers.clear()
    ordered = [all_d[i] for i in sorted(all_d)]
    return dict(base_report, status='updated', optimizer_step=optimizer.global_step,
        trained_rollouts=len(ordered), masked_rollouts=len(rows) - len(ordered),
        sampled_tokens=sum(len(r['rollout'].tokens) for r in rows),
        trained_tokens=sum(r['tokens'] for r in ordered),
        loss_tokens=sum(r['loss_tokens'] for r in ordered),
        derailed_rollouts=sum(r['derail_index'] is not None for r in ordered),
        unresolved_trained_rollouts=sum(r['outcome'] == 'unresolved' for r in ordered),
        outcome_counts={outcome: sum(r['rollout'].outcome == outcome for r in rows)
                        for outcome in ('correct', 'wrong', 'unresolved')},
        objective_loss=math.fsum(all_l[i] for i in sorted(all_l)), preclip_gradient_norm=norm,
        maximum_old_actor_logprob_drift=0.0, trajectories=ordered,
        optimizer=update, preparation_seconds=timings['gradient_seconds'],
        optimizer_seconds=time.monotonic() - step_started,
        update_seconds=time.monotonic() - started,
        microbatch=dict(engine='packed-zero3-v835', token_budget=dp.BUDGET, max_rows=dp.MAX_ROWS, count=nbins, world=world,
                        rank_tokens=loads, steps=job['steps'], logit_chunk=LOGIT_CHUNK,
                        sharding='fsdp-fully-shard-reshard-after-forward', probe_tokens=sum(length(i) for i in probe),
                        ranks=ranks),
        unresolved_filter=unresolved_filter,
        data_sha256=digest([dataclasses.asdict(group) for group in groups]),
        loss_aggregation=aggregation,
        epoch_metrics=B.epoch_metrics(rows, all_d, config, aggregation, len(active), norm))


def helper_main(rank, world, port, dp_path, source_root):
    """Ranks 1..W-1 of the sharded world: join both groups, then serve jobs until rank 0 says stop."""
    host_dir = str(Path(dp_path).resolve().parent)
    sys.path.insert(0, source_root); sys.path.insert(0, host_dir)
    os.environ[ZERO3_ENV] = '1'
    parent = os.getppid()

    def watchdog():   # the trainer (rank 0) is our parent: never outlive it
        while True:
            time.sleep(5)
            if os.getppid() != parent:
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ['MASTER_PORT'] = str(port)
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)   # CUDA_VISIBLE_DEVICES pins one GPU per helper
            dist.init_process_group('nccl', rank=rank, world_size=world, device_id=torch.device('cuda', 0),
                                    timeout=timedelta(minutes=45))
        else:
            dist.init_process_group('gloo', rank=rank, world_size=world, timeout=timedelta(minutes=45))
        ctrl = dist.new_group(backend='gloo', timeout=timedelta(days=30))

        def load(name, rel):
            spec = importlib.util.spec_from_file_location(name, os.path.join(host_dir, rel)); m = importlib.util.module_from_spec(spec)
            sys.modules[name] = m; spec.loader.exec_module(m); return m
        dp = load('v83_dp', 'v83_dp.py')
        B = load('v824_batched', 'v8211_batched.py'); FK = load('v83_fast_kernels', 'v83_fast_kernels.py')
        KP = load('v83_kernels_policy', 'v83_kernels_policy.py')
        KP.allow()
        dp.STATE.update(world=world, ctrl=ctrl, initialized=True)
        dp.install_fast_hash()
        make_train_update(dp, B, FK)      # installs the sharded forward on this rank's v824_batched
        _event('helper-zero3', rank=rank, world=world, backend=_backend())
    except BaseException as x:
        _fatal(rank, 'start', x)
    while True:
        job = dp._bcast(None)
        if job is None or job.get('op') == 'stop':
            break
        try:
            B.DROP_UNRESOLVED, B.LOSS_AGG_MODE, B.LOSS_SCALE = job['drop'], job['mode'], job['scale']
            B.SUPPORT_RENORM = job.get('support')
            B.KNOBS = job.get('knobs')
            if WORLD['actor'] is None:
                WORLD['actor'] = _load_actor(B, FK, dp, job, world, rank)
                _event('helper-sharded', rank=rank, shard_bytes=sum(WORLD['actor'].shard_bytes().values()))
            config = job['config']; agg_config, loss_scale, _ = B.loss_aggregation(config)
            groups = job['groups']
            rows = B.layout_drop_unresolved(groups, config)[0] if B.DROP_UNRESOLVED else B._layout(groups, config)
            active = [i for i, row in enumerate(rows) if row['active']]
        except BaseException as x:
            _fatal(rank, 'sharding', x)
        errors, details, losses, info = _run(B, dp, job, rows, rank, config=config, agg_config=agg_config,
                                             loss_scale=loss_scale, active_rollouts=len(active),
                                             guard=lambda: None, record=lambda row: None)
        if errors:
            continue
        gathered = [None] * world
        dist.all_gather_object(gathered, (details, losses, {i: (rows[i]['keep'], rows[i]['derail_index']) for i in details}, info),
                               group=ctrl)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == '__main__':
    if len(sys.argv) == 7 and sys.argv[1] == 'helper':
        helper_main(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
    else:
        raise SystemExit('usage: v835_zero3.py helper RANK WORLD PORT V83_DP_PY SOURCE_ROOT')
