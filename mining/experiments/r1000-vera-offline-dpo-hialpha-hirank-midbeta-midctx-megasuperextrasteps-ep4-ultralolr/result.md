# R1000 — p4138 MERGE→n80 armed

**Status:** MERGE_READY (16 shards) → lean chall `:8003` GPUs 4,5 on `mine-r338` → v4 n80 vs reign36.

**Axis:** vera Soft→MidCtx HiAlpha HiRank MidBeta Mega UltraLoLR (β=0.1 r=64 lr=5e-7 @8192 steps=19200).

**Check:** `ssh … -p 40099` → `tail -f /root/logs/p4138_r1000_chall_n80_wvk7.log` · result `/root/affine_data/r1000_sim_result_reign36_wvk7.json`.

**Decision rule (pre-registered):** submit-license iff margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 under wvk=7 k=3 τ=0.03.
