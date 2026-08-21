# R1105 — ShortCtx HiRank Midβ Hyper HiLR

**Status (p4235):** **TRAIN LIVE** on `mine-r252` GPUs 6,7 after R1081 REFUTE reap.

| field | value |
|---|---|
| parent | R1081 Ultra HiLR REFUTE ~0.40× → HyperExtra isolate (28800→38400) |
| knobs | β=0.1 α=128 r=64 lr=2e-6 @6144 Soft Mid Mid Soft→ShortCtx **max_steps=38400** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
