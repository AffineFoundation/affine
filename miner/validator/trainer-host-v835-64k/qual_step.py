#!/usr/bin/env python3
"""Synthetic forward+backward+optimizer qualification for 32K / 64K on 8xH200.

Designed for ``torchrun --nproc_per_node=8``. Builds synthetic rollouts (no real
miner data), runs v834c or v835 against a Qwen3.5-class model, times the real
CPU AdamW / fresh-adam-normgrad boundary, and writes per-rank peak CUDA memory
+ step timings to JSON.

Modes:

  * ``singleton`` — one (prompt, output) per rank (default; stress long context)
  * ``packed`` — 125 groups × 8 rollouts with mixed lengths → several micro-batches

Fails the process (non-zero) if **any** rank reports an exception or OOM.

Dependencies (operator-installed on the H200 box):

  * torch >= 2.14 with CUDA
  * transformers 5.14.x (Qwen3_5ForConditionalGeneration)
  * flash-attn / torch.nn.attention.varlen
  * flash-linear-attention (fla) 0.6.x
  * **triton >= 3.7.1** (3.7.0 breaks fla gated ``chunk_bwd`` on Hopper)
  * safetensors / accelerate as required by transformers

This script does **not** touch live wallets, R2 credentials, or production pods.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# Per-rank Triton cache BEFORE importing torch/triton-using modules (Hopper fla).
_local = os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0'))
os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(
    os.environ.get('TRITON_CACHE_ROOT', '/tmp/triton-cache-v835'), f'rank{_local}'))

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load(name: str, rel: str):
    path = HERE / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def qwen35_9b_config():
    """Public geometry matching the live SN120 actor (REPORT.md)."""
    from transformers import Qwen3_5Config
    layer_types = (['linear_attention'] * 3 + ['full_attention']) * 8
    return Qwen3_5Config(
        vocab_size=248320,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        max_position_embeddings=262144,
        layer_types=layer_types,
        torch_dtype='bfloat16',
    )


def build_model(model_spec: str, device: torch.device):
    from transformers import Qwen3_5ForConditionalGeneration
    if model_spec.startswith('random://'):
        cfg = qwen35_9b_config()
        model = Qwen3_5ForConditionalGeneration(cfg)
    else:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_spec, torch_dtype=torch.bfloat16, trust_remote_code=False, use_safetensors=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    for name, p in model.named_parameters():
        active = name.startswith('model.language_model.') or name.startswith('lm_head.')
        p.requires_grad_(active)
    return model


def install_checkpoint(model):
    from torch.utils.checkpoint import checkpoint
    layers = model.model.language_model.layers
    for layer in layers:
        original = layer.forward

        def make_forward(original):
            def forward(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(original, *args, use_reentrant=False, **kwargs)
                return original(*args, **kwargs)
            return forward

        layer.forward = make_forward(original)


def synthetic_batch(prompt_len: int, output_len: int, vocab: int, seed: int = 120):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed + output_len)
    prompt = torch.randint(1, min(vocab, 10000), (prompt_len,), generator=g).tolist()
    output = torch.randint(1, min(vocab, 10000), (output_len,), generator=g).tolist()
    return prompt, output


def packed_plan(group_count: int, group_size: int, prompt_len: int, output_lens: list,
                vocab: int, token_budget: int, world: int, rank: int):
    """Build a 125×8-style synthetic batch and LPT-assign bins to this rank."""
    rows = []
    seed = 9001
    for g in range(group_count):
        for n in range(group_size):
            out_len = output_lens[(g * group_size + n) % len(output_lens)]
            prompt, output = synthetic_batch(prompt_len, out_len, vocab, seed=seed + g * 17 + n)
            rows.append(dict(prompt=prompt, output=output, length=len(prompt) + len(output)))
    # First-fit-decreasing bins then LPT across ranks (mirrors v835_dp._plan).
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]['length'], i))
    bins = []
    for i in order:
        placed = False
        for b in bins:
            if b[0] + rows[i]['length'] <= token_budget:
                b[0] += rows[i]['length']; b[1].append(i); placed = True; break
        if not placed:
            bins.append([rows[i]['length'], [i]])
    loads = [0] * world
    assign = [[] for _ in range(world)]
    for cost, members in sorted(bins, key=lambda b: (-b[0], b[1])):
        r = min(range(world), key=lambda k: loads[k])
        loads[r] += cost
        assign[r].append(sorted(members))
    return rows, assign[rank], dict(nbins=len(bins), loads=loads, local_bins=len(assign[rank]))


def packed_selected_loss(model, FK, items, logit_chunk: int, support: dict | None):
    class _RT:
        def __init__(self, model, device, vocab):
            self.model = model
            self.device = device
            self.vocab_size = vocab
            self.max_context_tokens = 69632
            self.trainable = True

        def guard(self):
            return None

        def _tokens(self, xs):
            return list(xs)

    if FK is None:
        raise RuntimeError('fast kernels required for qual_step packed loss')
    rt = _RT(model, next(model.parameters()).device, model.config.vocab_size)
    # Drive thresholded activation offload the same way as v835_dp.fast_blp.
    try:
        import v835_dp as dp
        tokens = sum(len(p) + len(o) for p, o in items)
        dp.STATE['offload_this_forward'] = bool(
            dp.ACTIVATION_OFFLOAD and tokens >= dp.ACTIVATION_OFFLOAD_MIN_TOKENS)
    except Exception:
        pass
    try:
        out = FK.packed_log_probs(rt, items, gradients=True, logit_chunk=logit_chunk, support=support)
    finally:
        try:
            import v835_dp as dp
            dp.STATE['offload_this_forward'] = False
        except Exception:
            pass
    losses = []
    if support is None:
        for log_p in out:
            losses.append(-log_p.mean())
    else:
        for log_p, log_q, inside in out:
            pg = torch.where(inside, log_q, torch.zeros_like(log_q))
            losses.append(-(pg.mean() + 0.01 * log_p.mean()))
    return torch.stack(losses).sum()


def memory_snapshot(rank: int) -> dict:
    if not torch.cuda.is_available():
        return dict(rank=rank, cuda=False)
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    return dict(
        rank=rank,
        cuda=True,
        device=torch.cuda.current_device(),
        device_name=torch.cuda.get_device_name(),
        max_memory_allocated=int(torch.cuda.max_memory_allocated()),
        max_memory_reserved=int(torch.cuda.max_memory_reserved()),
        memory_allocated=int(torch.cuda.memory_allocated()),
        memory_reserved=int(torch.cuda.memory_reserved()),
        active_bytes_all_peak=int(stats.get('active_bytes.all.peak', 0)),
        allocated_bytes_all_peak=int(stats.get('allocated_bytes.all.peak', 0)),
        reserved_bytes_all_peak=int(stats.get('reserved_bytes.all.peak', 0)),
    )


def _normgrad_cpu_step(named_params, gradients, lr=5e-7, cap=1.0, tensor_rel_cap=2e-5):
    """Production-shaped fresh-adam normgrad on CPU FP32 masters (no host secrets)."""
    live = [(n, p) for n, p in named_params if p.requires_grad]
    masters = {n: p.detach().float().cpu().clone() for n, p in live}
    qd, gn2, r, wnd = {}, {}, {}, {}
    with torch.no_grad():
        for name, _ in live:
            g = gradients[name].detach().float().cpu()
            w32 = masters[name]
            # fresh-adam delta ≈ lr * g / (|g| + eps) with mcore-ish wd skipped for speed here;
            # normgrad uses ||d|| from that delta and direction g/||g||.
            eps = 1e-8
            d = lr * (g / (g.abs() + eps))
            qd[name] = float(d.double().pow(2).sum())
            gn2[name] = float(g.double().pow(2).sum())
            wn = math.sqrt(float(w32.double().pow(2).sum())); wnd[name] = wn
            rt = 1.0
            if tensor_rel_cap is not None and qd[name] > 0 and wn > 0:
                rt = min(1.0, tensor_rel_cap * (1 - 1e-6) * wn / math.sqrt(qd[name]))
            r[name] = rt
        raw = math.sqrt(math.fsum(qd[n] * r[n] * r[n] for n, _ in live))
        scale = min(1.0, cap / raw) if raw > 0 else 1.0
        changed = 0
        for name, p in live:
            g = gradients[name].detach().float().cpu()
            c = (scale * r[name] * math.sqrt(qd[name]) / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0
            masters[name].sub_(g * c)
            new = masters[name].to(dtype=p.dtype)
            changed += int((new.to(p.device) != p).sum())
            p.copy_(new.to(device=p.device))
    return dict(raw_update_norm=raw, cap_scale=scale, projected_changed_elements=changed,
                implementation='qual-normgrad-cpu-fp32-master-v1')


def _publication_stub(named_params):
    """Cheap stand-in for post-step publication hashing (no R2 / secrets)."""
    h = hashlib.sha256()
    for name, p in named_params:
        if not p.requires_grad:
            continue
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return dict(bf16_sha256=h.hexdigest(), publication='local-digest-stub')


def run_variant(variant, model, microbatches, logit_chunk, support, timings_out):
    """``microbatches`` is a list of item-lists ``[(prompt, output), ...]`` for this rank."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    FK = _load('v83_fast_kernels', 'v83_fast_kernels.py')
    FK.install(model)
    dp = _load('v835_dp', 'v835_dp.py')
    if variant == 'v835':
        class _R:
            pass
        rt = _R(); rt.model = model; rt.trainable = True
        dp._install_activation_offload(rt)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.monotonic()
    oom = False
    err = None
    norm = None
    phase = {}

    try:
        for p in model.parameters():
            p.grad = None
        if variant == 'v834c':
            from subnet.mimo_grpo.optimizer import FP32Accumulator
            acc = FP32Accumulator(model.named_parameters())
            t = time.monotonic()
            for items in microbatches:
                if not items:
                    # Idle rank still participates in collectives later with zero contrib.
                    continue
                loss = packed_selected_loss(model, FK, items, logit_chunk, support)
                loss.backward()
                acc.capture()
            torch.cuda.synchronize(); phase['gradient_seconds'] = time.monotonic() - t
            t = time.monotonic()
            for name in sorted(acc.buffers):
                dist.reduce(acc.buffers[name], dst=0, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(); phase['reduce_seconds'] = time.monotonic() - t
            phase['gather_seconds'] = 0.0
            if rank == 0:
                t = time.monotonic()
                norm = acc.clip(1.0)
                phase['clip_seconds'] = time.monotonic() - t
                t = time.monotonic()
                # Pageable GPU→CPU as production CPUAdamW.step does.
                cpu_grads = {n: g.detach().to('cpu', copy=True) for n, g in acc.buffers.items()}
                phase['grads_to_host_seconds'] = time.monotonic() - t
                t = time.monotonic()
                opt_info = _normgrad_cpu_step(acc.parameters, cpu_grads)
                phase['optimizer_seconds'] = time.monotonic() - t
                t = time.monotonic()
                pub = _publication_stub(acc.parameters)
                phase['publication_seconds'] = time.monotonic() - t
                phase['optimizer'] = opt_info; phase['publication'] = pub
            else:
                phase['clip_seconds'] = phase['grads_to_host_seconds'] = 0.0
                phase['optimizer_seconds'] = phase['publication_seconds'] = 0.0
            acc.buffers.clear()
        elif variant == 'v835':
            from v835_sharded_accum import ShardedFP32Accumulator
            acc = ShardedFP32Accumulator(model.named_parameters(), rank=rank, world=world,
                                         device=next(model.parameters()).device,
                                         max_inflight=dp.REDUCE_MAX_INFLIGHT)
            acc.install_hooks()
            # Lock-step micro-batches across ranks (pad with dummy backward).
            counts = [None] * world
            dist.all_gather_object(counts, len(microbatches))
            max_n = max(counts)
            padded = list(microbatches) + [None] * (max_n - len(microbatches))
            t = time.monotonic()
            for items in padded:
                for p in model.parameters():
                    p.grad = None
                if items:
                    loss = packed_selected_loss(model, FK, items, logit_chunk, support)
                    loss.backward()
                else:
                    loss = None
                    for _, p in model.named_parameters():
                        if p.requires_grad:
                            loss = p.float().sum() * 0.0 if loss is None else loss + p.float().sum() * 0.0
                    loss.backward()
                acc.capture()
            acc.finish()
            torch.cuda.synchronize(); phase['gradient_seconds'] = time.monotonic() - t
            phase['reduce_seconds'] = 0.0  # overlapped into backward hooks
            t = time.monotonic()
            norm = acc.clip_shards(1.0)
            torch.cuda.synchronize(); phase['clip_seconds'] = time.monotonic() - t
            t = time.monotonic()
            full = acc.gather_to_cpu(dst=0)
            torch.cuda.synchronize(); phase['gather_seconds'] = time.monotonic() - t
            phase['grads_to_host_seconds'] = 0.0  # gather already pinned CPU
            if rank == 0:
                t = time.monotonic()
                opt_info = _normgrad_cpu_step(acc.parameters, full)
                phase['optimizer_seconds'] = time.monotonic() - t
                t = time.monotonic()
                pub = _publication_stub(acc.parameters)
                phase['publication_seconds'] = time.monotonic() - t
                phase['optimizer'] = opt_info; phase['publication'] = pub
            else:
                phase['optimizer_seconds'] = phase['publication_seconds'] = 0.0
            acc.clear()
        else:
            raise ValueError('variant must be v834c or v835')
    except torch.cuda.OutOfMemoryError as e:
        oom = True
        err = 'CUDA OOM: ' + str(e).split('\n')[0][:240]
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
        traceback.print_exc()

    phase['step_seconds'] = time.monotonic() - t0
    phase['preclip_norm'] = norm
    snap = memory_snapshot(rank)
    payload = dict(rank=rank, variant=variant, oom=oom, error=err, memory=snap, **phase,
                   microbatches=len(microbatches),
                   items=sum(len(b) for b in microbatches),
                   tokens=sum(len(p) + len(o) for b in microbatches for p, o in b))
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    timings_out.extend(gathered)
    return gathered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('v834c', 'v835'), required=True)
    p.add_argument('--mode', choices=('singleton', 'packed'), default='singleton')
    p.add_argument('--output-len', type=int, default=65536)
    p.add_argument('--prompt-len', type=int, default=4096)
    p.add_argument('--model', default='random://qwen3_5_9b')
    p.add_argument('--logit-chunk', type=int, default=4096)
    p.add_argument('--group-count', type=int, default=125)
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--token-budget', type=int, default=69632)
    p.add_argument('--json-out', required=True)
    args = p.parse_args()

    # Fail fast on known-bad Triton before spending minutes on model init.
    try:
        import triton
        ver = getattr(triton, '__version__', '0')
        parts = [int(x) for x in ver.split('.')[:3] if x.isdigit()]
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts) < (3, 7, 1):
            raise RuntimeError(f'triton>={3}.{7}.{1} required for fla gated chunk_bwd on Hopper; found {ver}')
    except ImportError:
        pass

    dist.init_process_group('nccl')
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        print(json.dumps(dict(event='qual-start', variant=args.variant, mode=args.mode,
                              output_len=args.output_len, model=args.model,
                              triton_cache=os.environ.get('TRITON_CACHE_DIR')), flush=True)

    model = build_model(args.model, device)
    install_checkpoint(model)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    support = dict(temperature=1.0, top_k=20, top_p=0.95)
    plan_meta = None
    if args.mode == 'singleton':
        prompt, output = synthetic_batch(args.prompt_len, args.output_len, model.config.vocab_size)
        microbatches = [[(prompt, output)]]
    else:
        # Mixed lengths so first-fit packing creates several micro-batches under the budget.
        output_lens = [2048, 4096, 8192, 12288, 16384, 24576, 32768]
        rows, local_bins, plan_meta = packed_plan(
            args.group_count, args.group_size, min(args.prompt_len, 512), output_lens,
            model.config.vocab_size, args.token_budget, world, rank)
        microbatches = [[(rows[i]['prompt'], rows[i]['output']) for i in members] for members in local_bins]
        if not microbatches:
            microbatches = [[]]

    reports = []
    run_variant(args.variant, model, microbatches, args.logit_chunk, support, reports)

    any_error = any(r.get('error') for r in reports)
    any_oom = any(r.get('oom') for r in reports)
    if rank == 0:
        summary = dict(
            variant=args.variant,
            mode=args.mode,
            model=args.model,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            logit_chunk=args.logit_chunk,
            world=world,
            plan=plan_meta,
            ranks=reports,
            any_oom=any_oom,
            any_error=any_error,
            max_allocated_bytes=max(r['memory'].get('max_memory_allocated', 0) for r in reports),
            max_reserved_bytes=max(r['memory'].get('max_memory_reserved', 0) for r in reports),
            triton_cache_dir_rank0=os.environ.get('TRITON_CACHE_DIR'),
        )
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        print(json.dumps(dict(event='qual-done', json_out=args.json_out, any_oom=any_oom,
                              any_error=any_error,
                              max_allocated_gib=summary['max_allocated_bytes'] / 1024**3)), flush=True)
        if any_error:
            print(json.dumps(dict(event='qual-FAIL', errors=[r.get('error') for r in reports if r.get('error')])),
                  flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if any_error:
        sys.exit(2)


if __name__ == '__main__':
    main()
