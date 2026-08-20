# R1017 — MidCtx MidRank Hiβ Ultra MidLR

**Status (p4150):** **TRAIN** on `mine-r338` GPUs 6,7 after R1007 REFUTE ~0.76×. MERGE→n80 armed.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.3** / 128 / **32** / **1e-6** |
| max_len / steps | **8192** MidCtx / **28800** Ultra |
| parent | R1007 Mega MidLR REFUTE m=+0.006593 ~0.76× |
