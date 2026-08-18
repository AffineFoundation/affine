# R722 — marsplan Soft MidRank MidBeta MidCtx SuperExtra ep3×LoLR
- base: marsplan0624/affine-5gedzafcvg-queen@556d02a2
- method: Offline DPO · β=0.1 · α=128 · r=32 · lr=1e-6 · @8192 · steps=14400 · ep=3
- decision: Stage-5 iff v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign34
- parent: after R708/R709 SoftCtx UltraExtra REFUTE; MidCtx β-sweep sibling of R721 LoBeta
