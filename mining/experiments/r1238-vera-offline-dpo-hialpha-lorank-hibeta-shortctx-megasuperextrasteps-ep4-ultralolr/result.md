# R1238 — ShortCtx LoRank Hiβ Mega UltraLoLR

- **Axis:** vera×Offline-DPO×HiAlpha×LoRank×HiBeta×ShortCtx×Mega×ep4×UltraLoLR
- **Base:** `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- **Knobs:** β=0.3 α=128 r=16 lr=5e-7 @6144 max_steps=19200
- **Parent:** R1217 ShortCtx LoRank Hiβ Mega MidLR REFUTE m=+0.004629 ~0.89× (thought✓ B✓) → Mega UltraLoLR isolate (Mega HiLR already R1230 TRAIN)
- **Pod:** mine-crown-1 / brave-comet-f4 GPUs 4,5
- **Decision:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
- **p4339:** TRAIN after exact-PID R1217 chall reap
