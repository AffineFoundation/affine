# R1145 — ShortCtx LoRank Midβ Hyper UltraLoLR

**Status (p4286):** **REFUTE v4** vs reign36 → **R1166** MidLR.

| field | value |
|---|---|
| parent | R1127 ShortCtx LoRank Midβ Hyper HiLR REFUTE m=−0.001642 ~−0.25× → UltraLoLR isolate |
| knobs | β=0.1 α=128 r=**16** lr=**5e-7** @6144 Soft Mid Mid Soft→ShortCtx **max_steps=38400** |
| n80 | m=**−0.001512** SE=0.002307 z=−0.655 n=78 bar≈0.004613 (~**−0.33×**) thought✓(166) B✓(0.428) k=3/τ=0.03 |
| decision | REFUTE — margin < bar (thought/B cleared) |

## p4285 (2026-08-21T08:55Z)
- Triton wipe+seed + `/v1/completions` probe → PROBE_OK → n80 LIVE.

## p4286 (2026-08-21T09:02Z)
- Harvest REFUTE; exact-PID reap :8003 GPUs 6,7 → **R1166** MidLR TRAIN pid **75354**.
