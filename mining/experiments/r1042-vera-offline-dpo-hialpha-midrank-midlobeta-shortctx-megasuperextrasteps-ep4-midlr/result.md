# R1042 — ShortCtx MidRank MidLoβ Mega MidLR

**Status (p4184):** **v4 n80 LIVE** on `mine-crown-1` :8002 pid**219093** vs reign36.
Train DONE @18:13Z (884 steps) → MERGE `/tmp/r1042_merged` → CHALL_READY → n80 launched 18:18:16Z.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **1e-6** |
| max_len / steps | **6144** / **19200** (ran ~884) |
| parent | R1035 ShortCtx MidRank MidLoβ Ultra MidLR REFUTE m=−0.006514 ~−0.63× → Mega MidLR isolate |
| out | `/root/affine_data/r1042_sim_result_reign36_wvk7.json` |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
