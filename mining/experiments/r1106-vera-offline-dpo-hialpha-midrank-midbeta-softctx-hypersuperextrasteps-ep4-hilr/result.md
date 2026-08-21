# R1106 — SoftCtx MidRank Midβ Hyper HiLR

**Status (p4253):** **REFUTE v4** vs reign36.

| field | value |
|---|---|
| parent | R1083 SoftCtx MidRank Midβ Hyper MidLR REFUTE ~0.53× → Hyper HiLR isolate |
| knobs | β=0.1 α=128 r=32 lr=2e-6 @12288 Soft Mid Mid Soft **max_steps=38400** |
| n80 | m=**+0.002112** SE=0.003619 z=0.584 n=79 bar≈0.007239 (~**0.29×**) thought✓(195) B✓(0.477) k=3/τ=0.03 |
| decision | Stage-5 iff v4 n80 margin>max(2·SE,δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
| next | → **R1125** SoftCtx LoRank Midβ Hyper HiLR (r=16) on freed r337 GPUs 6,7 |
