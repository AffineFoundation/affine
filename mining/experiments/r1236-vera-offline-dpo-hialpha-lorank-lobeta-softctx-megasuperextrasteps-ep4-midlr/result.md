# R1236 — SoftCtx LoRank Loβ Mega MidLR

- **Axis:** vera×Offline-DPO×HiAlpha×LoRank×LoBeta×SoftCtx×Mega×ep4×MidLR
- **Base:** `vera6/affine-5g4yy75zuz-t6`@`8e3f1695`
- **Knobs:** β=0.02 α=128 r=16 lr=1e-6 @12288 max_steps=19200
- **Parent:** R1165 MidCtx LoRank Loβ Hyper MidLR ~0.39× + R1226 MidCtx Mega MidLR TRAIN → SoftCtx LoRank Loβ Mega MidLR isolate
- **Pod:** mine-r1191-vera-fullft-1 / swift-comet-4d GPUs 4,5
- **Decision:** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
- **p4339:** TRAIN launched on idle GPUs
