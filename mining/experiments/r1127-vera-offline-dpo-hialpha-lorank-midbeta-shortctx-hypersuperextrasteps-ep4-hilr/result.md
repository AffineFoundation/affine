# R1127 — ShortCtx LoRank Midβ Hyper HiLR

**Status (p4254):** **TRAIN LIVE** on `mine-r339` GPUs 6,7 after R1109 REFUTE reap.

| field | value |
|---|---|
| parent | R1109 ShortCtx HiRank Hiβ Hyper HiLR REFUTE ~0.17× → LoRank+Midβ isolate |
| knobs | β=0.1 α=128 r=**16** lr=2e-6 @6144 Soft Mid Mid Soft→ShortCtx **max_steps=38400** |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
