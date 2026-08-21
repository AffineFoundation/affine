# R1191 — result notes

## Status
- **p4307:** rented **8×H200** `mine-r1191-vera-fullft-1` / `swift-comet-4d` @$32/h by **node id** `4eb39f3b-3714-42f0-93eb-f81238c07834` (`eager-fox-11`). API+ssh **ngpu=8**. TTL→**2026-08-22T11:53Z**. **BOOTSTRAP_DONE** + **FullFT TRAIN LIVE** pid**2434** GPUs**0–7** (lr=1e-6 ep=1 @8192 thought-only). Teacher DL + post_train armed. SSH `69.63.236.163:40299`.

## Axis
vera6 reign36 × FullFT thought-only dense SFT (≠ Offline-DPO LoRA fleet; ≠ Online-DPO marsplan; ≠ R1158 Reason-GRPO). Base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`.

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.
