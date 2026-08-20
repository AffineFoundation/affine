# R1050 — SoftCtx MidRank Hiβ Ultra HiLR

Parent: **R1039 REFUTE** m=+2.37e-5 ~0.006× (thought✓170 B✓0.481 k=3).

Isolates **HiLR** (lr=`2e-6`) on SoftCtx MidRank Hiβ Ultra after MidLR flop.

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / r / α | 0.3 / 32 / 128 |
| lr / max_len / steps | **2e-6** / 12288 / 28800 |
| pod | `mine-r338-…` GPUs **6,7** :8002 |

## Timeline
- p4180: TRAIN pid**133331** + MERGE→n80 waiter
- p4195: MERGE✓ (`/tmp/r1050_merged` 16 shards) but first chall died — Triton REUSE of partial `chall_r1050` (missing `__triton_launcher.so`). FORCE wipe+seed from `chall_r978` (n_so=26) → chall pid**143658** CHALL_READY + probe_ok → **v4 n80 LIVE** sim pid**146119** @**2026-08-20T19:39:09Z**.
- p4196: **REFUTE v4** m=**−0.003658** SE=0.004164 z=−0.878 n=79 bar≈0.008328 (~**−0.44×**) thought✓(189) B✓(0.530) k=3/τ=0.03 vs reign36 → exact-PID reap :8002 → **R1065**.

Decision rule: Stage-5 iff fresh v4 n80 margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36.
