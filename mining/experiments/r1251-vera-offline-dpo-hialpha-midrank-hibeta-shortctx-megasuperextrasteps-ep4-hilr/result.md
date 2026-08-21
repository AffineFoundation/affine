# R1251 — ShortCtx MidRank Hiβ Mega HiLR
- base: vera6/affine-5g4yy75zuz-t6@8e3f1695
- method: Offline-DPO Mega max_steps=19200 epochs=4 α=128
- knobs: β=0.3 r=32 lr=2e-6 max_len=6144
- hardware: mine-r1214 brave-shark-4d 8×H200 GPU 7 (single; multi-GPU unsupported)
- signal: Short MidRank Midβ Mega Hi=R1227 TRAIN; Soft MidRank Hiβ Mega Hi=R1240 TRAIN → Short MidRank Hiβ Mega HiLR isolate; cell never created
- decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36
- status: TRAIN LIVE p4344
