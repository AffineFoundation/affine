# R1177 — TRAIN LIVE p4294
Axis: vera Offline-DPO HiAlpha LoRank Hiβ MidCtx HyperSuperExtra ep4 UltraLoLR
β=0.3 α=128 r=16 lr=5e-7 @8192 max_steps=38400
Parent: R1165 MidCtx LoRank Loβ MidLR REFUTE m=+0.001335 SE=0.001711 ~0.39× thought✓180 B✓0.443
Pod: mine-r338 calm-fox-6a SSH 95.133.253.90:40099 · TRAIN pid=228481 GPUs 6,7 + MERGE→n80 waiters
Decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
