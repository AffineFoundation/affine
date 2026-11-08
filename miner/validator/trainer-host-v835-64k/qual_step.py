#!/usr/bin/env python3
"""One synthetic forward+backward+optimizer step for 32K / 64K H200 qualification.

Designed for ``torchrun --nproc_per_node=8`` on an 8xH200 box. Builds a synthetic
rollout batch (no real miner data), runs either the v834c full-accumulator path or
the v835 ZeRO-2 sharded path against a Qwen3.5-class model, and writes per-rank
peak CUDA memory + step timings to a JSON file.

Model sources (pick one via ``--model``):

  * HuggingFace id of a public Qwen3.5 / MiMo checkpoint with the same hybrid
    architecture (preferred when weights are already cached).
  * ``random://qwen3_5_9b`` — construct ``Qwen3_5ForConditionalGeneration`` from a
    hard-coded 9B-like config and random BF16 init (no download; still needs the
    installed ``transformers`` build that provides ``Qwen3_5ForConditionalGeneration``).

Dependencies (operator-installed on the H200 box):

  * torch >= 2.14 with CUDA
  * transformers 5.14.x (Qwen3_5ForConditionalGeneration)
  * flash-attn / torch.nn.attention.varlen (for full-attention layers)
  * flash-linear-attention (fla) 0.6.x (causal_conv1d + chunk_gated_delta_rule)
  * accelerate / safetensors as required by transformers

This script does **not** touch live wallets, R2 credentials, or production pods.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

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
    # Hybrid pattern: (linear_attention x3 + full_attention) * 8 = 32 layers.
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
    # Match production: only text tower + lm_head trainable.
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


def synthetic_batch(prompt_len: int, output_len: int, vocab: int, device: torch.device):
    """One packed (prompt, output) pair as integer token lists + tensors for a toy loss."""
    g = torch.Generator(device='cpu')
    g.manual_seed(120 + output_len)
    prompt = torch.randint(1, min(vocab, 10000), (prompt_len,), generator=g).tolist()
    output = torch.randint(1, min(vocab, 10000), (output_len,), generator=g).tolist()
    return prompt, output


def packed_selected_loss(model, FK, prompt, output, logit_chunk: int, support: dict | None):
    """Use the same packed_log_probs path as production when FK is available; else toy head."""
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

    if FK is not None:
        rt = _RT(model, next(model.parameters()).device, model.config.vocab_size)
        items = [(prompt, output)]
        out = FK.packed_log_probs(rt, items, gradients=True, logit_chunk=logit_chunk, support=support)
        if support is None:
            log_p = out[0]
            return -log_p.mean()
        log_p, log_q, inside = out[0]
        pg = torch.where(inside, log_q, torch.zeros_like(log_q))
        return -(pg.mean() + 0.01 * log_p.mean())

    # Fallback without fla: plain HF forward (slow / may OOM differently; for smoke only).
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + output], device=device)
    logits = model(ids, use_cache=False).logits.float()[0]
    # Score output tokens only.
    targets = torch.tensor(output, device=device)
    idx = torch.arange(len(prompt) - 1, len(prompt) + len(output) - 1, device=device)
    selected = logits.index_select(0, idx)
    log_p = selected.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(selected, dim=-1)
    return -log_p.mean()


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


def run_variant(variant: str, model, prompt, output, logit_chunk: int, support, activation_offload: bool):
    rank = dist.get_rank()
    world = dist.get_world_size()
    FK = None
    try:
        FK = _load('v83_fast_kernels', 'v83_fast_kernels.py')
        FK.install(model)
    except Exception as e:
        if rank == 0:
            print(json.dumps(dict(event='fast-kernels-unavailable', error=repr(e)[:300])), flush=True)

    if variant == 'v835' and activation_offload:
        dp = _load('v835_dp', 'v835_dp.py')
        dp.ACTIVATION_OFFLOAD = True
        # Minimal runtime-like object for the offload installer.
        class _R:
            pass
        rt = _R()
        rt.model = model
        rt.trainable = True
        dp._install_activation_offload(rt)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.monotonic()
    oom = False
    err = None
    norm = None
    opt_seconds = None
    grad_seconds = None

    try:
        if variant == 'v834c':
            from subnet.mimo_grpo.optimizer import FP32Accumulator
            for p in model.parameters():
                p.grad = None
            acc = FP32Accumulator(model.named_parameters())
            t_grad = time.monotonic()
            loss = packed_selected_loss(model, FK, prompt, output, logit_chunk, support)
            loss.backward()
            acc.capture()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            grad_seconds = time.monotonic() - t_grad
            # SUM-reduce full buffers to rank 0 (v834c behaviour).
            for name in sorted(acc.buffers):
                dist.reduce(acc.buffers[name], dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                norm = acc.clip(1.0)
                t_opt = time.monotonic()
                # Stateless SGD stand-in (same boundary as host optimizer.step(acc.buffers)).
                with torch.no_grad():
                    for name, p in acc.parameters:
                        g = acc.buffers[name]
                        p.add_(g.to(dtype=p.dtype), alpha=-1e-7)
                opt_seconds = time.monotonic() - t_opt
            acc.buffers.clear()
        elif variant == 'v835':
            from v835_sharded_accum import ShardedFP32Accumulator
            for p in model.parameters():
                p.grad = None
            acc = ShardedFP32Accumulator(model.named_parameters(), rank=rank, world=world,
                                         device=next(model.parameters()).device)
            acc.install_hooks()
            t_grad = time.monotonic()
            loss = packed_selected_loss(model, FK, prompt, output, logit_chunk, support)
            loss.backward()
            acc.capture()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            grad_seconds = time.monotonic() - t_grad
            acc.reduce_shards()
            full = acc.gather_full_buffers(dst=0)
            if rank == 0:
                norm = acc.clip(1.0, buffers=full)
                t_opt = time.monotonic()
                with torch.no_grad():
                    for name, p in acc.parameters:
                        g = full[name]
                        p.add_(g.to(dtype=p.dtype), alpha=-1e-7)
                opt_seconds = time.monotonic() - t_opt
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
        if rank == 0:
            traceback.print_exc()

    total = time.monotonic() - t0
    snap = memory_snapshot(rank)
    # All-gather reports.
    payload = dict(
        rank=rank,
        variant=variant,
        oom=oom,
        error=err,
        preclip_norm=norm,
        gradient_seconds=grad_seconds,
        optimizer_seconds=opt_seconds,
        step_seconds=total,
        memory=snap,
        prompt_len=len(prompt),
        output_len=len(output),
        context_len=len(prompt) + len(output),
    )
    gathered = [None] * world
    dist.all_gather_object(gathered, payload)
    return gathered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=('v834c', 'v835'), required=True)
    p.add_argument('--output-len', type=int, required=True, help='synthetic output tokens (32768 or 65536)')
    p.add_argument('--prompt-len', type=int, default=4096)
    p.add_argument('--model', default='random://qwen3_5_9b')
    p.add_argument('--logit-chunk', type=int, default=4096)
    p.add_argument('--activation-offload', action='store_true')
    p.add_argument('--support-renorm', action='store_true', default=True)
    p.add_argument('--json-out', required=True)
    args = p.parse_args()

    dist.init_process_group('nccl')
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    rank = dist.get_rank()

    if rank == 0:
        print(json.dumps(dict(event='qual-start', variant=args.variant, output_len=args.output_len,
                              model=args.model, activation_offload=args.activation_offload)), flush=True)

    model = build_model(args.model, device)
    install_checkpoint(model)
    # Broadcast identical random init.
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    prompt, output = synthetic_batch(args.prompt_len, args.output_len, model.config.vocab_size, device)
    support = dict(temperature=1.0, top_k=20, top_p=0.95) if args.support_renorm else None

    # Only rank 0 holds the long synthetic tokens for the singleton-bin case; other ranks
    # participate in collectives with an empty local contribution (zero grads via a tiny
    # dummy loss on a 1-token sequence) when world>1 and we want a true single-bin 64K test.
    # For qualification of the worst-case singleton, every rank still builds the same length
    # so each GPU exercises the 64K activation footprint (matches "exercise every rank").
    reports = run_variant(args.variant, model, prompt, output, args.logit_chunk, support,
                          args.activation_offload)

    if rank == 0:
        summary = dict(
            variant=args.variant,
            model=args.model,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            logit_chunk=args.logit_chunk,
            activation_offload=args.activation_offload,
            support_renorm=bool(support),
            world=dist.get_world_size(),
            ranks=reports,
            any_oom=any(r['oom'] for r in reports),
            max_allocated_bytes=max(r['memory'].get('max_memory_allocated', 0) for r in reports),
            max_reserved_bytes=max(r['memory'].get('max_memory_reserved', 0) for r in reports),
        )
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        print(json.dumps(dict(event='qual-done', json_out=args.json_out,
                              any_oom=summary['any_oom'],
                              max_allocated_gib=summary['max_allocated_bytes'] / 1024**3)), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
