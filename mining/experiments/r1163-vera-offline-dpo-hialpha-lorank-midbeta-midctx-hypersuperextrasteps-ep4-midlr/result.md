# R1163 — MidCtx LoRank Midβ Hyper MidLR (after R1143 UltraLoLR REFUTE ~0.61×)

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.

**Status (p4299):** **REFUTE v4** vs reign36 → slot → **R1184** MidCtx MidRank Midβ UltraLoLR.

| field | value |
|---|---|
| margin | **+0.000438** SE=0.001202 z=0.365 n=78 bar≈0.002404 (~**0.18×**) |
| thought / B | ✓162 / ✓0.430 (k=3 τ=0.03) |
| knobs | β=0.1 α=128 r=16 lr=1e-6 @8192 Soft Mid Mid Soft→MidCtx **max_steps=38400** |
| next | MidCtx LoRank Midβ LR exhausted → **R1184** MidCtx MidRank Midβ Hyper **UltraLoLR** (R1107 HiLR ~0.66× / R1094 MidLR already) |

Parent: R1143 UltraLoLR m=+0.002751 SE=0.002264 bar=0.004528 ~0.61× thought✓172 B✓0.456 → MidLR isolate (1e-6).
