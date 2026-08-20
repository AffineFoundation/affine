# R985 result — v4 n80 vs reign36

**Pass:** p4130 · **Pod:** mine-r924-vera-midctx-hibeta-1 · chall :8002 GPUs 4,5

## Verdict
**REFUTE** (challenger_wins=false)

| metric | value |
|---|---|
| margin | **+0.002374** |
| SE | 0.002613 |
| z | 0.909 |
| bar | max(2·SE, δ) ≈ **0.005225** |
| margin/bar | **~0.45×** |
| n_paired | 79 |
| thought median | **199** ✓ (≥80) |
| B pass | **0.456** ✓ (≥0.30) |
| k / τ | 3 / 0.03 |

## Axis
vera Offline-DPO HiAlpha HiRank Midβ MidCtx UltraSuperExtraSteps ep4 **HiLR** (lr=2e-6, r=64, β=0.1, @8192, max_steps=28800)

## Follow-up
→ **R1003** MidCtx HiRank Midβ Ultra **UltraLoLR** (lr=5e-7, max_steps=28800) TRAIN p4130 on same GPUs 4,5.
