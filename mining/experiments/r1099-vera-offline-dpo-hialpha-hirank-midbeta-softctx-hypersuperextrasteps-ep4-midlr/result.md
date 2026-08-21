# R1082 → R1099 (p4230)

## R1082 n80 (REFUTE)
- vs reign36 vera `@8e3f1695`, wvk=7 k=3 τ=0.03
- margin=+0.000172 SE=0.004629 z=0.037 n=77 bar≈0.009258 (~**0.019×**)
- thought✓187 B✓0.405 — floors clear, margin fail
- axis: ShortCtx HiRank Midβ HyperExtra MidLR (β=0.1 r=64 α=128 lr=1e-6 @6144 steps=38400)

## Decision
SoftCtx isolate on same HiRank Midβ Hyper MidLR lane → **R1099**.

## R1099 plan
- base `vera6/affine-5g4yy75zuz-t6@8e3f1695`
- Offline-DPO Soft Mid Mid Soft · SoftCtx `@12288` · HiRank r=64 · Midβ=0.1 · α=128 · MidLR 1e-6 · Hyper steps=38400 · ep=4
- pod `mine-r938` GPUs 2,3 · MERGE→n80 waiters armed
- Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30
