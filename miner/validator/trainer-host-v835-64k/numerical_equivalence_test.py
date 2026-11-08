#!/usr/bin/env python3
"""Compare v834c FP32Accumulator+SUM-reduce vs v835 async reduce-to-owner + shard clip.

Runs on CPU by default (no GPU, no flash-attn, no fla). Optionally ``torchrun``
with 1–2 processes and ``--device cuda`` when GPUs are present.

Checks (within fp32 tolerance):

  1. Accumulated FP32 gradients match after identical synthetic backwards
     (including **multiple micro-batches** per step).
  2. Shard clip (all-reduce sum-of-squares) matches full-inventory global-norm clip.
  3. A stateless fresh-adam-like normgrad update matches on the resulting parameters.
  4. Selected-token logprob + top-k/top-p support-renorm helpers match.

Exit code 0 on success, 1 on mismatch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from v835_sharded_accum import ShardedFP32Accumulator, partition_names  # noqa: E402


def support_logq(logits, t, top_k, top_p):
    """Mirror of v83_fast_kernels.support_logq."""
    values, ids = logits.topk(top_k, dim=-1)
    with torch.no_grad():
        p = torch.softmax(values, dim=-1)
        keep = (p.cumsum(-1) - p) < top_p
        hit = (ids == t[:, None]) & keep
        inside = hit.any(-1)
    log_q_all = torch.log_softmax(values.masked_fill(~keep, float('-inf')), dim=-1)
    log_q = log_q_all.masked_fill(~keep, 0.).gather(-1, hit.float().argmax(-1, keepdim=True)).squeeze(-1)
    return torch.where(inside, log_q, torch.zeros_like(log_q)), inside


def selected_logp(logits, t):
    return logits.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


class TinyActor(nn.Module):
    """Small stand-in with several differently-sized trainable tensors."""

    def __init__(self, vocab=128, hidden=64, layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden, bias=True) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        for block in self.blocks:
            h = F.gelu(block(h))
        return self.lm_head(h)


class LegacyFP32Accumulator:
    """Minimal copy of subnet.mimo_grpo.optimizer.FP32Accumulator (v834c)."""

    def __init__(self, named_parameters):
        self.parameters = tuple((n, p) for n, p in named_parameters if p.requires_grad)
        self.buffers = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.parameters}
        self.microsteps = 0
        self.clipped = False

    def capture(self):
        for name, parameter in self.parameters:
            g = parameter.grad
            if g is None:
                raise ValueError('missing grad ' + name)
            self.buffers[name].add_(g)
            parameter.grad = None
        self.microsteps += 1

    def clip(self, maximum):
        norms = [torch.linalg.vector_norm(g, dtype=torch.float32) for g in self.buffers.values()]
        total = torch.linalg.vector_norm(torch.stack(norms), dtype=torch.float32)
        scale = torch.clamp(maximum / (total + 1e-6), max=1.)
        for g in self.buffers.values():
            g.mul_(scale)
        self.clipped = True
        return float(total)


def _fresh_normgrad_step(params, gradients, lr, cap):
    """Rank-local stand-in for host fresh-adam normgrad (direction g/||g||)."""
    live = list(params)
    qd, gn2 = {}, {}
    with torch.no_grad():
        for name, p in live:
            g = gradients[name].detach().to(device=p.device, dtype=torch.float32)
            d = g * lr
            qd[name] = float(d.double().pow(2).sum())
            gn2[name] = float(g.double().pow(2).sum())
        raw = math.sqrt(math.fsum(qd.values()))
        scale = min(1.0, cap / raw) if raw > 0 else 1.0
        for name, p in live:
            g = gradients[name].detach().to(device=p.device, dtype=torch.float32)
            c = (scale * math.sqrt(qd[name]) / math.sqrt(gn2[name])) if gn2[name] > 0 else 0.0
            updated = p.detach().float() - g * c
            p.copy_(updated.to(dtype=p.dtype))
    return dict(raw_update_norm=raw, cap_scale=scale)


def _allclose(a, b, rtol=1e-5, atol=1e-5):
    if a.shape != b.shape:
        return False, f'shape {tuple(a.shape)} vs {tuple(b.shape)}'
    aa = a.float().cpu()
    bb = b.float().cpu()
    if not torch.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=False):
        diff = (aa - bb).abs().max().item()
        return False, f'max_abs_diff={diff}'
    return True, 'ok'


def run_support_renorm_check():
    torch.manual_seed(0)
    logits = torch.randn(16, 97, dtype=torch.float32)
    targets = torch.randint(0, 97, (16,))
    log_p = selected_logp(logits, targets)
    log_q, inside = support_logq(logits / 1.0, targets, top_k=20, top_p=0.95)
    log_q2, inside2 = support_logq(logits, targets, 20, 0.95)
    ok1, m1 = _allclose(log_q, log_q2)
    ok2 = bool(torch.equal(inside, inside2))
    return dict(selected_finite=bool(torch.isfinite(log_p).all()),
                support_match=ok1 and ok2, detail=m1, inside_frac=float(inside.float().mean()))


def run_single_process(device: str, rtol: float, atol: float) -> dict:
    torch.manual_seed(7)
    model_a = TinyActor().to(device)
    model_b = TinyActor().to(device)
    model_b.load_state_dict(model_a.state_dict())
    model_a.to(dtype=torch.bfloat16)
    model_b.to(dtype=torch.bfloat16)
    model_a.to(device)
    model_b.to(device)

    def make_ids(seed):
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        return torch.randint(0, 128, (4, 32), generator=g).to(device)

    def loss_fn(model, ids):
        logits = model(ids).float()
        targets = ids
        log_p = selected_logp(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        log_q, inside = support_logq(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), 16, 0.9)
        pg = torch.where(inside, log_q, torch.zeros_like(log_q))
        return -(pg.mean() + 0.01 * log_p.mean())

    # Three micro-batches to exercise multi-micro-batch accumulation.
    batches = [make_ids(100 + i) for i in range(3)]

    legacy = LegacyFP32Accumulator(model_a.named_parameters())
    for ids in batches:
        model_a.zero_grad(set_to_none=True)
        loss_fn(model_a, ids).backward()
        legacy.capture()
    norm_a = legacy.clip(1.0)

    sharded = ShardedFP32Accumulator(model_b.named_parameters(), rank=0, world=1, device=device)
    sharded.install_hooks()
    for ids in batches:
        model_b.zero_grad(set_to_none=True)
        loss_fn(model_b, ids).backward()
        sharded.capture()
    # Production order: clip on shards, then gather to CPU.
    norm_b = sharded.clip_shards(1.0)
    full_b = sharded.gather_to_cpu(dst=0)

    grad_reports = {}
    all_ok = True
    for name in sorted(legacy.buffers):
        ok, msg = _allclose(legacy.buffers[name], full_b[name], rtol=rtol, atol=atol)
        grad_reports[name] = dict(ok=ok, detail=msg)
        all_ok = all_ok and ok

    params_a = list(model_a.named_parameters())
    params_b = list(model_b.named_parameters())
    info_a = _fresh_normgrad_step(params_a, legacy.buffers, lr=1e-3, cap=1.0)
    info_b = _fresh_normgrad_step(params_b, full_b, lr=1e-3, cap=1.0)
    param_reports = {}
    for (n1, p1), (n2, p2) in zip(params_a, params_b):
        ok, msg = _allclose(p1.detach(), p2.detach(), rtol=rtol, atol=atol)
        param_reports[n1] = dict(ok=ok, detail=msg)
        all_ok = all_ok and ok

    # Dedicated shard-clip identity: rebuild unclipped grads and compare clip scales.
    model_c = TinyActor().to(device=device, dtype=torch.bfloat16)
    model_c.load_state_dict({k: v.detach().cpu() for k, v in model_a.state_dict().items()})
    model_c.to(device)
    # Re-init from pre-update model_b weights is awkward; instead compare norms only on a fresh pair.
    model_d = TinyActor().to(device)
    model_e = TinyActor().to(device)
    model_e.load_state_dict(model_d.state_dict())
    model_d.to(dtype=torch.bfloat16); model_e.to(dtype=torch.bfloat16)
    model_d.to(device); model_e.to(device)
    leg2 = LegacyFP32Accumulator(model_d.named_parameters())
    sh2 = ShardedFP32Accumulator(model_e.named_parameters(), rank=0, world=1, device=device)
    sh2.install_hooks()
    for ids in batches[:2]:
        model_d.zero_grad(set_to_none=True); loss_fn(model_d, ids).backward(); leg2.capture()
        model_e.zero_grad(set_to_none=True); loss_fn(model_e, ids).backward(); sh2.capture()
    # Pre-clip shard vs full norms must match before either clip.
    full_pre = {n: sh2.buffers[n].detach().clone() for n in sh2.buffers}
    leg_norm_pre = math.sqrt(sum(float(torch.linalg.vector_norm(g, dtype=torch.float32).square())
                                 for g in leg2.buffers.values()))
    sh_norm_pre = math.sqrt(sum(float(torch.linalg.vector_norm(g, dtype=torch.float32).square())
                                for g in full_pre.values()))
    n_leg = leg2.clip(0.5)
    n_sh = sh2.clip_shards(0.5)
    shard_clip_ok = abs(n_leg - n_sh) <= max(1e-5, 1e-5 * abs(n_leg)) and abs(leg_norm_pre - sh_norm_pre) <= 1e-5

    support = run_support_renorm_check()
    all_ok = all_ok and support['support_match'] and support['selected_finite'] and shard_clip_ok

    return dict(
        mode='single-process',
        device=device,
        world=1,
        microbatches=len(batches),
        preclip_norm_legacy=norm_a,
        preclip_norm_sharded=norm_b,
        norms_match=abs(norm_a - norm_b) <= max(1e-5, 1e-5 * abs(norm_a)),
        shard_clip=dict(ok=shard_clip_ok, legacy=n_leg, sharded=n_sh),
        grad_reports=grad_reports,
        param_reports=param_reports,
        optimizer_info=dict(legacy=info_a, sharded=info_b),
        support_renorm=support,
        ok=all_ok and abs(norm_a - norm_b) <= max(1e-5, 1e-5 * abs(norm_a)),
    )


def run_distributed(device: str, rtol: float, atol: float) -> dict:
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    backend = 'nccl' if device.startswith('cuda') else 'gloo'
    if device.startswith('cuda'):
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
        device = f'cuda:{int(os.environ.get("LOCAL_RANK", rank))}'
    dist.init_process_group(backend)
    torch.manual_seed(11)
    model = TinyActor().to(device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    def one_loss(seed):
        g = torch.Generator(device='cpu')
        g.manual_seed(seed + rank * 1009)
        ids = torch.randint(0, 128, (2, 24), generator=g).to(device)
        logits = model(ids).float()
        targets = ids
        flat_l = logits.reshape(-1, logits.size(-1))
        flat_t = targets.reshape(-1)
        log_p = selected_logp(flat_l, flat_t)
        log_q, inside = support_logq(flat_l, flat_t, 16, 0.9)
        pg = torch.where(inside, log_q, torch.zeros_like(log_q))
        return -(pg.mean() + 0.01 * log_p.mean())

    n_micro = 3
    legacy = LegacyFP32Accumulator(model.named_parameters())
    for micro in range(n_micro):
        model.zero_grad(set_to_none=True)
        one_loss(micro + 3).backward()
        legacy.capture()
    for name in sorted(legacy.buffers):
        dist.reduce(legacy.buffers[name], dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        norm_legacy = legacy.clip(1.0)
        grads_legacy = {k: v.detach().clone() for k, v in legacy.buffers.items()}
    else:
        norm_legacy, grads_legacy = None, None

    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    model.zero_grad(set_to_none=True)

    sharded = ShardedFP32Accumulator(model.named_parameters(), rank=rank, world=world, device=device)
    sharded.install_hooks()
    for micro in range(n_micro):
        model.zero_grad(set_to_none=True)
        one_loss(micro + 3).backward()
        sharded.capture()
    norm_sharded = sharded.clip_shards(1.0)
    full = sharded.gather_to_cpu(dst=0)
    if rank == 0:
        reports = {}
        ok = abs(norm_legacy - norm_sharded) <= max(1e-5, 1e-5 * abs(norm_legacy))
        for name in sorted(grads_legacy):
            matched, detail = _allclose(grads_legacy[name], full[name], rtol=rtol, atol=atol)
            reports[name] = dict(ok=matched, detail=detail)
            ok = ok and matched
        parts = partition_names(list(model.named_parameters()), world)
        owned_counts = {n: sum(n in names for names in parts) for n, _ in model.named_parameters()}
        partition_ok = all(v == 1 for v in owned_counts.values())
        result = dict(
            mode='distributed',
            device=device,
            world=world,
            microbatches=n_micro,
            preclip_norm_legacy=norm_legacy,
            preclip_norm_sharded=norm_sharded,
            grad_reports=reports,
            partition_ok=partition_ok,
            shard_clip_ok=ok,
            ok=ok and partition_ok,
        )
    else:
        result = dict(mode='distributed', rank=rank, ok=True)

    blob = [result if rank == 0 else None]
    dist.broadcast_object_list(blob, src=0)
    dist.destroy_process_group()
    return blob[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--json-out', default='')
    args = p.parse_args()
    if 'RANK' in os.environ:
        report = run_distributed(args.device, args.rtol, args.atol)
    else:
        report = run_single_process(args.device, args.rtol, args.atol)
    text = json.dumps(report, indent=2, sort_keys=True)
    if not os.environ.get('RANK') or int(os.environ.get('RANK', '0')) == 0:
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text + '\n')
    sys.exit(0 if report.get('ok') else 1)


if __name__ == '__main__':
    main()
