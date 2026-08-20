# R1016 — MidCtx HiRank Midβ Ultra MidLR

**Status (p4166):** **REFUTE v4** vs reign36 on `mine-r924` GPUs 4,5.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **64** / **1e-6** |
| max_len / steps | **8192** MidCtx / **28800** Ultra |
| parent | R1003 Ultra UltraLoLR REFUTE ~0.40× |

**n80 (wvk7 k=3 τ=0.03):** m=**+0.002671** SE=0.002993 z=0.892 n=79 bar≈0.005987 (~**0.45×**) thought✓(170) B✓(0.532). wins=false.

**Next:** R1033 MidCtx HiRank MidLoβ Ultra MidLR (β 0.1→0.05 isolate).
