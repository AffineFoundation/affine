# R1247 — MidCtx HiRank Midβ Mega HiLR
- base: vera6/affine-5g4yy75zuz-t6@8e3f1695
- method: Offline-DPO Mega max_steps=19200 epochs=4 α=128
- knobs: β=0.1 r=64 lr=2e-6 max_len=8192
- hardware: mine-r1214 brave-shark-4d 8×H200 GPU 3 (single; multi-GPU unsupported)
- signal: R1231 Short HiRank Midβ Mega HiLR TRAIN + R1204 Short MidLR REFUTE → MidCtx Mega HiLR isolate; cell missing
- decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36
- status: TRAIN LIVE p4343
