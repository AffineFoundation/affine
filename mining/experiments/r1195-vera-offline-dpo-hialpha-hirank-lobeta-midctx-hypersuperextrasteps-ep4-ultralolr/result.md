# R1195 — TRAIN LIVE p4310

Axis: vera Offline-DPO HiAlpha HiRank Loβ MidCtx HyperSuperExtra ep4 UltraLoLR
β=0.02 α=128 r=64 lr=5e-7 @8192 max_steps=38400
Parent: R1180 MidCtx HiRank Loβ MidLR REFUTE m=+0.001514 SE=0.003187 ~0.24×;
SoftCtx HiRank Loβ LR exhausted (R1135/R1164/R1151); MidCtx HiLR R1122 → MidCtx UltraLoLR isolate
Pod: mine-r338 calm-fox-6a SSH 95.133.253.90:40099 · TRAIN pid=243160 GPUs 4,5 + MERGE→n80 waiters
Decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,δ=0.002) AND thought≥80 AND B≥0.30 vs reign36
