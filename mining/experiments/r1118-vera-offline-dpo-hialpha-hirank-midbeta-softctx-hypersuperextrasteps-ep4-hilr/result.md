# R1099 → R1118 (p4248)

## R1099 n80 (REFUTE)
- vs reign36 vera `@8e3f1695`, wvk=7 k=3 τ=0.03
- margin=−0.001230 SE=0.001555 z=−0.791 n=80 bar≈0.003109 (~**−0.40×**)
- thought✓179.5 B✓0.4375 — floors clear, margin fail
- axis: SoftCtx HiRank Midβ HyperExtra MidLR (β=0.1 r=64 α=128 lr=1e-6 @12288 steps=38400)

## Decision
Hyper HiLR isolate on same SoftCtx HiRank Midβ HyperExtra lane → **R1118**.

## R1118 plan
- base `vera6/affine-5g4yy75zuz-t6@8e3f1695`
- Offline-DPO Soft Mid Mid Soft · SoftCtx `@12288` · HiRank r=64 · Midβ=0.1 · α=128 · HiLR 2e-6 · Hyper steps=38400 · ep=4
- pod `mine-r938` GPUs 2,3 · MERGE→n80 waiters armed
- Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30
