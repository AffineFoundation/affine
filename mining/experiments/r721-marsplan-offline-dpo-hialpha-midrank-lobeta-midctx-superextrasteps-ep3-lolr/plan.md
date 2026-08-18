# R721 — marsplan Soft MidRank LoBeta MidCtx SuperExtra ep3×LoLR

**Axis:** transfer R637 Soft MidRank LoBeta SIGNAL (~1.45×) / R692 MidCtx MidRank LoBeta UltraExtra onto **marsplan** with MidCtx + SuperExtra steps.

| knob | value |
|---|---|
| base | `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2` |
| method | Offline-DPO · HiAlpha |
| β | 0.02 (LoBeta) |
| LoRA | r=32 · α=128 |
| lr | 1e-6 · ep=3 |
| max_len | 8192 (MidCtx) |
| max_steps | 14400 (SuperExtra) |
| data | MidCtx MidRank LoBeta (from r692; 604 pairs) |
| GPUs | lunar **6,7** |

**≠** R712 SoftCtx SuperExtra marsplan · ≠ R692 UltraExtra r252 · ≠ R719 MidBeta MidCtx r252 · ≠ R698 HyperExtra MidBeta MidCtx · ≠ Online / ≠ GRPO

**Decision:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign34 (k=3, τ=0.03).
