# R1127 — ShortCtx LoRank Midβ Hyper HiLR

**Status (p4270):** **N80 LIVE** on `mine-r339` TP1 util**0.85** :8003 GPU6 (prior TP2 chall died mid-tokenizer; merge reused).

| field | value |
|---|---|
| knobs | β=0.1 α=128 r=**16** lr=2e-6 @6144 Soft Mid Mid Soft→ShortCtx **max_steps=38400** |
| chall / n80 | pid**59089** / pid**61316** · SSH `23.153.44.20:40299` |
| check | `tail -f /root/logs/p4270_r1127_chall_n80_wvk7.log` · progress `/root/affine_data/r1127_sim_progress_reign36_wvk7.json` |
| decision | Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 |
