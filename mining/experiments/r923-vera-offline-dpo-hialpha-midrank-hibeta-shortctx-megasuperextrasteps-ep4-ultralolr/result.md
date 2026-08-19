# R923 — vera ShortCtx Hiβ UltraLoLR

**Status:** **n80 RUNNING p4036** R888 :8002 CHALL_READY · sim pid**42129** → `/root/affine_data/r923_sim_result_reign36_wvk7.json`
**Axis:** Soft Mid Mid Soft MidRank HiBeta ShortCtx UltraLoLR (β=0.3 r=32 α=128 lr=5e-7 @6144 ep4 steps=19200)
**Parent:** R914 ShortCtx Loβ ~0.14× → Hiβ isolate
**Ops note p4035:** prior wait used `--adapter /root/r923/train` (peft miss); fixed → `/root/r923/train/adapter`; MERGE_DONE 16 shards; Triton seeded
**Ops note p4036:** chall READY 20:18:55Z · n80 launched 20:18:56Z block_hash=`bf97f76a…` vs reign36 vera
**Decision rule:** Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36
