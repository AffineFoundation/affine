# R1106 — SoftCtx MidRank Midβ Hyper HiLR

**Status (p4236):** **TRAIN LIVE** on `mine-r337` GPUs 6,7 after R1083 REFUTE reap.

| field | value |
|---|---|
| parent | R1083 SoftCtx MidRank Midβ Hyper MidLR REFUTE ~0.53× → Hyper HiLR isolate |
| knobs | β=0.1 α=128 r=32 lr=2e-6 @12288 Soft Mid Mid Soft **max_steps=38400** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
