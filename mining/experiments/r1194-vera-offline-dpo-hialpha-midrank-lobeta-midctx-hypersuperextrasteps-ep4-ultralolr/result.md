# R1194 — TRAIN LIVE p4310

Axis: vera Offline-DPO HiAlpha MidRank Loβ MidCtx HyperSuperExtra ep4 UltraLoLR
β=0.02 α=128 r=32 lr=5e-7 @8192 max_steps=38400
Parent: R1177 MidCtx LoRank Hiβ UltraLoLR REFUTE m=−0.002441 SE=0.002156 ~−0.57×;
MidCtx LoRank Hiβ LR exhausted (R1091 Mid / R1103 Hi / R1177 Ultra) → MidRank Loβ UltraLoLR MidCtx isolate
Pod: mine-r338 calm-fox-6a SSH 95.133.253.90:40099 · TRAIN pid=243122 GPUs 6,7 + MERGE→n80 waiters
Decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
