# R1235 — ShortCtx LoRank Midβ Mega MidLR

- **Axis:** vera×Offline-DPO×HiAlpha×LoRank×MidBeta×ShortCtx×Mega×ep4×MidLR
- **Base:** `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- **Knobs:** β=0.1 α=128 r=16 lr=1e-6 @6144 max_steps=19200
- **Parent:** R1166 ShortCtx LoRank Midβ Hyper MidLR REFUTE ~0.45× → Mega MidLR isolate (cell was missing)
- **Pod:** mine-r1191-vera-fullft-1 / swift-comet-4d GPUs 2,3
- **Decision:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
- **p4339:** TRAIN launched on idle GPUs (fill burn floor)
