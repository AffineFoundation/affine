# R923 — vera ShortCtx Hiβ UltraLoLR

**Status:** **n80 LOADING p4035** R888 :8002 chall pid**40190** (MERGE_DONE 16 shards; Triton seeded from chall_r914)
**Axis:** Soft Mid Mid Soft MidRank HiBeta ShortCtx UltraLoLR (β=0.3 r=32 α=128 lr=5e-7 @6144 ep4 steps=19200)
**Parent:** R914 ShortCtx Loβ ~0.14× → Hiβ isolate
**Ops note p4035:** prior wait used `--adapter /root/r923/train` (peft miss); fixed → `/root/r923/train/adapter` via `lean_merge_r888_gpus56_p4035.sh`; n80 waiter armed (`wait_r923_merge_then_n80_p4035.sh`)
**Decision rule:** Stage-5 iff fresh v4 n80 margin>max(2·SE,0.002) AND thought≥80 AND B≥0.30 vs reign36
