# R1158 — result notes

## Status
- **p4304:** BOOT_HF_DONE → teacher TP1 :8000 READY + **GRPO TRAIN LIVE** pid**5163** GPUs**2,3** (α=128 r=16 G=4 lr=5e-6 @6144 steps=1800). SSH `204.12.163.23:20301`.
- **p4303:** rented **8×H200** `mine-r1158-vera-reason-grpo-1` / `eager-matrix-57` @$32/h by **node id** `e350ebc9-8012-4789-ac47-7a573d70bf9a` (`golden-orbit-7b`). API+ssh **ngpu=8**. TTL→**2026-08-22T11:23Z**. Bootstrap LIVE then DONE.
- p4281: teacher DL done, still 1 GPU → tore brave-matrix; blind re-rent returned **same** exec **fbb1135f-cffe-4962-9389-150ec0e0852b**@192.9.163.79 (`eager-lion-45`, API `gpu_count=1`) → tore again. Lesson: **blind `lium up --gpu` ignores executor_blacklist**. Waiter now rents by **node id** + verifies **ngpu≥8** (ssh+API) before bootstrap.

## Axis
vera6 reign36 × Reason-GRPO HiAlpha (≠ Offline-DPO UltraLoLR fleet; ≠ Online-DPO marsplan). Base `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`.

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.
