# R1230 — ShortCtx LoRank Hiβ Mega HiLR

Parent: **R1203 REFUTE** m=−0.001873 SE=0.002267 z=−0.83 n=80 bar≈0.004533 (~**−0.41×**) thought✓188.5 B✓0.438 k=3/τ=0.03 vs reign36 (p4335). SoftCtx Mega HiLR=R1219 / Ultra=R1216 TRAIN; ShortCtx Mega MidLR=R1217 TRAIN → ShortCtx Mega HiLR isolate.

Axis: vera Offline-DPO HiAlpha LoRank Hiβ ShortCtx Mega ep4 **HiLR**
- base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- β=0.3 α=128 r=16 lr=**2e-6** @6144 max_steps=19200
- GPUs 3,4 on `mine-r340` after R1203 chall reap

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30.
