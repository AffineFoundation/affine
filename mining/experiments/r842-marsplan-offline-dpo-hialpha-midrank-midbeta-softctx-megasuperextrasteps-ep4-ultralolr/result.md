# R842 — marsplan SoftCtx MidRank MidBeta UltraLoLR

**Decision rule:** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, δ=0.002) AND median |z|≥80 AND B≥0.30 vs reign36 vera.

| field | value |
|---|---|
| base | `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2` |
| method | Offline DPO · Soft Mid Mid Soft data @12288 |
| knobs | β=0.1 α=128 r=32 lr=5e-7 ep=4 steps=19200 UltraLoLR |
| parent | R836 MidCtx Mid Mid REFUTE m=−0.00576 ~−0.85× → SoftCtx transfer |
| status | TRAIN p3928 lunar GPUs 6,7 |
