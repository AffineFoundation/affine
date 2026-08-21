# R1249 — SoftCtx LoRank Loβ Mega UltraLoLR
- base: vera6/affine-5g4yy75zuz-t6@8e3f1695
- method: Offline-DPO Mega max_steps=19200 epochs=4 α=128
- knobs: β=0.02 r=16 lr=5e-7 max_len=12288
- hardware: mine-r1214 brave-shark-4d 8×H200 GPU 5 (single; multi-GPU unsupported)
- signal: R1236 Soft LoRank Loβ Mega MidLR TRAIN + R1244 Soft LoRank Loβ Mega HiLR TRAIN → Mega UltraLoLR isolate; cell never created
- decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36
- status: TRAIN LIVE p4344
