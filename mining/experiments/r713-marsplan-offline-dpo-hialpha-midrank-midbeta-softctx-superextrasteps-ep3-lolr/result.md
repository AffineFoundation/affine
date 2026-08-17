# R713 — marsplan Soft MidRank MidBeta SoftCtx SuperExtra

## Claim
Offline-DPO on `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2` with Soft MidRank MidBeta SoftCtx data, β=0.1, r=32, α=128, lr=1e-6, max_len=12288, max_steps=14400, ep=3 clears next crown under Reason v4 (k=3, τ=0.03).

## Decision rule (pre-registered)
Stage-5 iff fresh v4 n80 paired margin > max(2·SE, δ=0.002) **and** median |z|≥80 **and** B pass≥0.30 vs reign34.

## Why this axis
Transfer R675 Soft MidRank MidBeta SoftCtx UltraExtra ~0.97× onto marsplan with SuperExtra steps; ≠ R708 UltraExtra 7200 / ≠ R702 HyperExtra REFUTE / ≠ r252 SuperExtra R710.

## Status (p3750)
**TRAIN** on `mine-r226-marsplan-fullft-1` (brave-raven-a9) GPUs **0,1** pid**257892** · wait→merge armed · kept=604 · freed ~660G old /tmp merges for headroom. Lesson: pin BASE after `mine.env` (brave overwrites BASE→r252).
