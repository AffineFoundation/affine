# R933 result — Reason v4 n80 vs reign36

**Axis:** cryptoDev23 × Offline-DPO × HiAlpha × MidRank × MidLoBeta × ShortCtx × MegaSuperExtra × ep4 × UltraLoLR
**Base:** `cryptoDev23/Affine-5Dku3dYp9j-hk8161`@`55b7ffe0`
**Knobs:** β=0.05 α=128 r=32 lr=5e-7 max_len=6144 steps=19200
**Pod:** mine-r337 noble-hawk-1f GPUs 6,7 :8003
**Stamp:** wvk=7 k=3 τ=0.03

## Verdict — REFUTE (p4053)

| metric | value |
|---|---|
| margin | **−0.005798** |
| SE | 0.004286 |
| z | −1.353 |
| n | 80 |
| bar = max(2·SE, δ) | ≈0.008571 |
| margin/bar | **~−0.68×** |
| thought median | **200** ✓ (≥80) |
| B pass | **0.45** ✓ (≥0.30) |
| challenger_wins | false |

## Notes
- Late host-relay kept piping after n80 LIVE; p4053 killed parent+children by exact PID + rm `*.tmp` to protect `/tmp/r933_merged` under vllm/sim.
- Slot freed → **R941** SoftCtx MidLoβ MegaSuperExtra on vera GPUs 6,7.
