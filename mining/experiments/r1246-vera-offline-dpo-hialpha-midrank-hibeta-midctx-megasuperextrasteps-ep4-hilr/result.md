# R1246 — MidCtx MidRank Hiβ Mega HiLR
- base: vera6/affine-5g4yy75zuz-t6@8e3f1695
- method: Offline-DPO Mega max_steps=19200 epochs=4 α=128
- knobs: β=0.3 r=32 lr=2e-6 max_len=8192
- hardware: mine-r1214 brave-shark-4d 8×H200 GPU 1 (single; multi-GPU unsupported)
- signal: R1240 Soft MidRank Hiβ Mega HiLR TRAIN sibling → MidCtx isolate; Soft MidRank Loβ Mega Hi=R1215 REFUTE; cell missing
- decision: Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36
- status: TRAIN LIVE p4343
