# R939 — vera Offline SoftCtx MidRank Midβ UltraExtraSteps

- Base: vera6/affine-5g4yy75zuz-t6 @ 8e3f1695 (reign36)
- Method: Offline DPO Soft Mid Mid Soft
- Knobs: β=0.1, r=32, α=128, lr=5e-7, max_len=12288, epochs=4, max_steps=28800
- Parent: R337 online-DPO HiLR REFUTE ~−0.95× → Offline SoftCtx Midβ UltraExtra isolate
- Decision: Stage-5 iff v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30
