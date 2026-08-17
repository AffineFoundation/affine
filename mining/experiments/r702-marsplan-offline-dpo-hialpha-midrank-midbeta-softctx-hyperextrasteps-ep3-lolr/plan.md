# R702 — marsplan Soft MidRank MidBeta SoftCtx HyperExtra ep3×LoLR

**Axis:** non-king marsplan × Offline-DPO × HiAlpha × MidRank × MidBeta × SoftCtx × HyperExtraSteps × epochs=3 × LoLR

| knob | value |
|---|---|
| base | `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2…` |
| β | 0.1 |
| LoRA r/α | 32 / 128 |
| lr | 1e-6 |
| max_len | 12288 |
| max_steps | **10800** |
| epochs | 3 |

**Why:** transfer R675 Soft MidRank MidBeta SoftCtx UltraExtra near-miss (~0.97×) onto marsplan with HyperExtra 1.5× steps. ≠ R701 marsplan Soft MidRank LoBeta SoftCtx HyperExtra. ≠ R699 r252 Soft MidRank MidBeta SoftCtx HyperExtra. ≠ R672 marsplan Soft MidRank LoBeta SoftCtx Mega.

**Decision rule:** Stage-5 iff fresh v4 n80 (k=3, τ=0.03) paired margin > max(2·SE, 0.002) AND median |z|≥80 AND B≥0.30 vs reign34.

**Pod:** zesty-comet-da GPUs 6,7 (idle after R691 REFUTE; R694 n80 owns 4,5).
