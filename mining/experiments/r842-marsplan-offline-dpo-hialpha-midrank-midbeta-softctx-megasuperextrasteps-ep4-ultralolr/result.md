# R842 — marsplan SoftCtx MidRank MidBeta UltraLoLR

**Decision rule:** Stage-5 iff fresh v4 n80 paired margin > max(2·SE, δ=0.002) AND median |z|≥80 AND B≥0.30 vs reign36 vera.

| field | value |
|---|---|
| base | `marsplan0624/affine-5gedzafcvg-queen`@`556d02a2` |
| method | Offline DPO · Soft Mid Mid Soft data @12288 |
| knobs | β=0.1 α=128 r=32 lr=5e-7 ep=4 steps=19200 UltraLoLR |
| parent | R836 MidCtx Mid Mid REFUTE m=−0.00576 ~−0.85× → SoftCtx transfer |
| status | **n80 RELOAD p3943** lunar :8003 GPUs 6,7 · MERGE_DONE · prior EngineDead mid-probe ~22:30Z |

**p3943:** Chall died on `sample_tokens` RPC timeout during probe; one-shot `r842_n80_launched.p3928` left GPUs idle. Cleared stamp + relaunched `lean_chall_n80_lunar_gpus67_p3928.sh` (pid in `/root/logs/p3943_r842_chall_n80.pid`).
