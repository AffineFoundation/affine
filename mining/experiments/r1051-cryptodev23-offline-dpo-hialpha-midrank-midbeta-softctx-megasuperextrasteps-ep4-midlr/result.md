# R1051 result — pass 4214

## Axis
cryptoDev23 × Offline-DPO × HiAlpha × MidRank × MidBeta × SoftCtx × MegaSuperExtraSteps × ep4 × MidLR
- base: `cryptoDev23/Affine-5Dku3dYp9j-hk8161`@`55b7ffe003d078a8a131673f677b2584548a502e`
- β=0.1 · α=128 · r=32 · lr=1e-6 · max_len=12288 · max_steps=19200

## Decision rule (pre-registered)
Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) **and** median |z|≥80 **and** B≥0.30 vs reign36.

## v4 n80 (p4212/p4214) vs reign36
| field | value |
|---|---|
| wins | **false** |
| margin | **+0.0015536** |
| SE | 0.002913 |
| z | 0.533 |
| n | 79 paired / 80 turns |
| bar | max(2·SE,δ) ≈ **0.005826** |
| × bar | **~0.27×** |
| thought median | 223 ✓ |
| B pass | 0.405 ✓ |
| k / τ | 3 / 0.03 ✓ |

## Verdict
**REFUTE.** Positive but far below bar. SoftCtx MidRank Midβ Mega MidLR on cryptoDev does not clear crown.

## Follow-on (same pass)
→ **R1080** cryptoDev ShortCtx MidRank Hiβ Ultra MidLR (R1032 crown-lane mirror) on `mine-r926` GPUs 3,4 after chall reap.
