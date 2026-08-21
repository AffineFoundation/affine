# R1237 — MidCtx LoRank Loβ Mega HiLR

- **Axis:** vera×Offline-DPO×HiAlpha×LoRank×LoBeta×MidCtx×Mega×ep4×HiLR
- **Base:** `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- **Knobs:** β=0.02 α=128 r=16 lr=2e-6 @8192 max_steps=19200
- **Parent:** R1226 MidCtx LoRank Loβ Mega MidLR TRAIN sibling → Mega HiLR isolate (cell was missing)
- **Pod:** mine-r1191-vera-fullft-1 / swift-comet-4d GPUs 6,7
- **Decision:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
- **p4339:** TRAIN launched on idle GPUs
