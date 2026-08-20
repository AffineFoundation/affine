# R1041 — MidCtx LoRank Midβ Mega HiLR

**Status (p4184):** **v4 n80 LIVE** on `mine-crown-1` :8004 pid**219229** vs reign36.
Train DONE @18:13Z (900 steps) → MERGE `/tmp/r1041_merged` → CHALL_READY → n80 launched 18:18:21Z.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **16** / **2e-6** |
| max_len / steps | **8192** / **19200** (ran ~900) |
| parent | R1031 MidCtx LoRank Midβ Ultra MidLR REFUTE m=+0.001897 ~0.32× → Mega HiLR isolate |
| out | `/root/affine_data/r1041_sim_result_reign36_wvk7.json` |

**Decision rule:** Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
