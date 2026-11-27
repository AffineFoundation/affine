#!/usr/bin/env python3
"""Static per-GPU memory estimator for SN120 9B GRPO at 32K / 64K (v834c vs v835).

Does not run a model. Uses the publicly known MiMo/Qwen3.5-9B geometry from the
2026-11-08 diagnosis: hidden 4096, 32 layers, vocab 248320, ~8.954B trainable
parameters (historical master archive ~35.815 GB FP32).

Activation term assumes checkpointing saves one BF16 input activation per block
(S x H) for all 32 blocks — the same bound used in REPORT.md. Workspaces
(FLA / FlashAttention / NCCL / allocator fragmentation) are NOT included; treat
the total as a lower bound on peak, not an exact measurement.
"""
from __future__ import annotations

import argparse
import json
import math


# Historical archive / REPORT.md inventory (approximate, decimal GB called out in docs).
TRAINABLE_NUMEL = 8_954_000_000  # ~8.954B
VOCAB = 248_320
HIDDEN = 4096
LAYERS = 32
# lm_head is the largest single trainable tensor (vocab x hidden).
LM_HEAD_NUMEL = VOCAB * HIDDEN


def giB(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


def estimate(seq_len: int, world: int = 8, logit_chunk: int = 4096) -> dict:
    bf16_w = TRAINABLE_NUMEL * 2
    fp32_full = TRAINABLE_NUMEL * 4
    shard = math.ceil(TRAINABLE_NUMEL / world) * 4
    # Checkpointed block inputs: 32 * S * H * 2 bytes
    acts = LAYERS * seq_len * HIDDEN * 2
    chunk_logits = logit_chunk * VOCAB * 4  # FP32 selected-head chunk
    lm_head_grad = LM_HEAD_NUMEL * 2

    v834c = dict(
        bf16_weights_gib=giB(bf16_w),
        fp32_accumulator_gib=giB(fp32_full),
        bf16_param_grads_gib=giB(bf16_w),
        saved_activations_gib=giB(acts),
        logit_chunk_fp32_gib=giB(chunk_logits),
        subtotal_static_gib=giB(bf16_w + fp32_full + bf16_w + acts),
        note='full FP32 accum + full BF16 grads coexist on every rank',
    )
    v835 = dict(
        bf16_weights_gib=giB(bf16_w),
        fp32_accumulator_shard_gib=giB(shard),
        bf16_param_grad_peak_gib=giB(lm_head_grad),  # one tensor in flight under hooks
        reduce_scratch_gib=giB(LM_HEAD_NUMEL * 4),
        saved_activations_gib=giB(acts),
        logit_chunk_fp32_gib=giB(chunk_logits),
        subtotal_static_gib=giB(bf16_w + shard + lm_head_grad + acts),
        with_activation_offload_static_gib=giB(bf16_w + shard + lm_head_grad),
        note='ZeRO-2 FP32 shard + post-accumulate BF16 disposal; workspaces excluded',
    )
    h200_gib = 139.80  # allocator-visible capacity from live OOM reports
    return dict(
        seq_len=seq_len,
        world=world,
        trainable_numel=TRAINABLE_NUMEL,
        h200_allocator_gib=h200_gib,
        v834c=v834c,
        v835=v835,
        expected_v834c_64k='OOM (static ~90+ GiB before workspaces vs ~140 GiB H200)',
        expected_v835_64k_headroom_gib=h200_gib - v835['subtotal_static_gib'],
        expected_v835_64k_with_offload_headroom_gib=h200_gib - v835['with_activation_offload_static_gib'],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seq', type=int, nargs='+', default=[36864, 69632],
                   help='prompt+output lengths to estimate')
    p.add_argument('--world', type=int, default=8)
    p.add_argument('--json-out', type=str, default='')
    args = p.parse_args()
    rows = [estimate(s, world=args.world) for s in args.seq]
    text = json.dumps(rows, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, 'w') as f:
            f.write(text)
            f.write('\n')


if __name__ == '__main__':
    main()
