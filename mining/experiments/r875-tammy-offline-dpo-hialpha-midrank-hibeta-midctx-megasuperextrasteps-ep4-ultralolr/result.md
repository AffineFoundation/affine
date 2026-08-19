# R875 — pass 3984

**Status:** TRAIN_DONE @2026-08-19T04:13:52Z (900 steps) → **MERGE LIVE** brave GPUs 4,5 (outer wait→TP1 chall+n80 armed).

**Axis:** tammy Soft Mid Mid Soft MidRank HiBeta MidCtx UltraLoLR (β=0.3 r=32 α=128 lr=5e-7 @8192 ep4).

**Decision rule (pre-registered):** Stage-5 iff fresh v4 n80 margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36 (fail-closed k=3).

**Next:** poll `/root/logs/r875_merge.done` → chall :8002 GPU4 TP1 → `r875_decision_reign36_wvk7.json`.
