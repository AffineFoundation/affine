# v8.2.9 trainer-speed: Step 1 profile (read-only), e81 job mimo-train-6d98b9c8...
Times UTC (CR = UTC-6). Sources: nvidia-smi every 3 s (nvsmi-e81.csv), host /proc sampler every 3 s (host-e81.txt),
training-events.jsonl observed_at, report.json. Samplers stopped 02:39 UTC.

## Machine
gentle-raven-de <TRAINER_HOST>:<TRAINER_SSH_PORT>: **1x NVIDIA H200 (141 GB HBM)**, 16 vCPU, 196 GB RAM. One GPU only; 139.6 GB
held all the time (actor bf16 + frozen KL reference + caching allocator, resident between jobs). torch 2.14 cu130,
transformers 5.14.1. **No flash_attn, no fla, no causal-conv1d** (transformers warns "fast path is not available ...
falling back to torch implementation"); attention=eager; engine=padded, token_budget 16384, max_rows 64.
Model qwen3_5_text: 32 layers = 24 Gated-DeltaNet linear-attention + 8 full-attention, hidden 4096, vocab 248320.
Optimizer: CPU AdamW fp32 masters (14 threads), ONE optimizer step per epoch (ppo_epochs 1, 256 groups x 8).

## e81 job phases (published 02:10:51, FINALIZED 02:40:40)
| phase | UTC | s | GPU util | GPU W | notes |
|---|---|---|---|---|---|
| pickup | 02:10:51-02:10:54 | 3 | 0 | 117 | |
| inputs + validation (256 docs fetched one by one, dispatch auth) | -02:12:49 | 115 | 0% | 117 | CPU 2%, rx 1.2 MB/s: latency-bound, model already resident |
| KL-reference pass (no grad) | -02:15:45 | 176 | 96% | 633 | 2.21M tokens -> 12.8k tok/s |
| grad micro-batches (164) | -02:32:20 | 995 | 93% | 575 (p10 484) | 2.21M loss tokens -> 2.2k tok/s; ~4.7 s per 12k-token short batch, ~10-12 s per 2x8k long batch; padding eff 0.94 |
| clip + CPU AdamW step | -02:33:00 | 40 | 3% | 128 | CPU 56%, 36.9 s |
| save + publish + optimizer-state write | -02:37:59 | 299 | 0% | 123 | model save ~60 s; model upload 18.4 GB at ~240 MB/s (80 s); then ~150 s disk-only fp32 optimizer state (~125 GB written at ~465 MB/s) |
| lifecycle readback/validate (Arbos) | -02:40:40 | 161 | 0% | | full SHA-256 GET of checkpoint |
Job total 27.1 min (1626 s) + 2.7 min finalize. GPU busy ~1170 s of 1626 s (72%); idle 456 s.
End-to-end 1.36k trained tok/s. Estimated MFU in grad phase ~12% (6*~9e9*2.2k/s vs 989 TF bf16 dense), ref pass ~25%:
"100% util" is long-running slow kernels (torch GDN fallback + eager attention), not saturation.
Same shape on e79/e80: update 1558/1383 s, ref 217/194 s, optimizer 36 s, publish ~300 s, inputs 154/120 s.

## (A) pure-throughput (same data, same single step, same lr; identical up to numerical tolerance)
1. fla Gated-DeltaNet + causal-conv1d kernels and SDPA/flash for the 8 full-attention layers, then the already-written
   `packed` varlen engine (no padding): grad+ref 1170 s -> est. 400-600 s. **Saves ~10-13 min.** Not bitwise; needs
   offline tolerance proof (per-token logprob, loss, grad-norm, update vs live e81 checkpoint). fla is pure Triton
   (Triton present); causal-conv1d is a CUDA build (fla has a Triton conv fallback).
2. Write the ~125 GB fp32 optimizer-state snapshot after the result is reported (before the next job; next job is >=30 min
   away) or to a faster non-FUSE disk. **Saves ~2.5 min.** Changes crash-durability ordering; needs a careful recovery design.
3. Fetch the 256 input documents in parallel (8-16 at a time; same bytes, same hashes). **Saves ~1.5 min.**
4. Lifecycle: start the publication readback as each model object lands (preupload), not after the report. **~2 min.**
   Lifecycle-side change -> separate signed release; touches the publisher/observer chain check (rebind needed).
5. Overlap CPU AdamW per-parameter with model shard save/upload. **~0.5-1 min.**
6. Multi-GPU data parallel with the same global batch: impossible on this 1-GPU pod. On an 8-GPU pod the grad+ref
   phase would drop ~6x (~16 min saved today) but needs a new/other machine (others' pods are off-limits).
Total realistic on this pod: ~27 min job -> ~12-14 min.

## (B) changes learning dynamics (proposal only, NOT implemented)
- 2-4 optimizer steps per epoch over the same 256 groups (mini-batches): needs an OLD-logprob pre-pass (+~3 min once,
  ~1 min after (A1)) and +37 s CPU AdamW per extra step. More learning per epoch, more off-policy.
- ppo_epochs 2 (reuse data twice): +1 full grad pass (~16 min now, ~6-8 min after A1).
- Larger global batch (max_tasks 256 -> 512): time ~ linear in tokens (+~17 min now, ~+7 min after A1); mining-limited.
- Learning rate 1e-6 -> 2e-6 (or schedule): no time cost; pure dynamics.
- Drop/shrink KL anchor (kl 0.02): -3 min now, changes objective.
- Lower overlong cap 8192: fewer tokens; changes reward/objective.

## Step 2 findings (02:58 UTC)
- A1 blocked by the signed runtime pin (see PROGRESS). A3 offline: sequential 63.0 s -> 8 threads 5.2 s (same bytes).
- Cold restart cost: first job after v8.2.5 drain-restart staged 11.7 min (18:04:51 -> 18:16:32 UTC) vs ~2 min warm.
