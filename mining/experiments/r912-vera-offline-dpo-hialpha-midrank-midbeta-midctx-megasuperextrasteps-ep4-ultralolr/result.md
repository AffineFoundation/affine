# R912 — vera MidCtx Midβ UltraLoLR

**Status:** **COLD→TRAIN p4026** after TTL-collapse lost prior crown train
**Axis:** Soft Mid Mid Soft MidRank MidBeta MidCtx UltraLoLR (β=0.1 r=32 α=128 lr=5e-7 @8192 ep4 steps=19200)
**Parent signal:** R901 MidCtx MidLoβ m=+0.002527~0.64× → Midβ isolate
**Pod:** mine-crown-1 GPUs 6,7 · bootstrap outer pid**1220**
**Data:** Soft Mid Mid Soft from local r886 (`dpo_duel_reason.jsonl`, 604 rows)
**Decision rule:** Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36

**Check:** `tail -f /root/logs/bootstrap_crown_r912_r913_p4026.log` → then `/root/logs/r912_train.nohup`
