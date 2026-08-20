# R1047 — MidCtx MidRank MidLoβ Ultra HiLR

**Status (p4192):** **v4 n80 LIVE** on `mine-r337` chall `:8003` GPUs4,5 after MERGE sat idle (waiter path bug).

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.05** / 128 / **32** / **2e-6** |
| ctx / steps | MidCtx `@8192` / Ultra `28800` |
| parent | R1030 REFUTE ~0.38× → Ultra HiLR isolate |

## Timeline
- p4179: TRAIN launched GPUs4,5 pid110242 + MERGE→n80 waiter
- 2026-08-20T19:09Z train.done · 19:13Z merge.done → `/tmp/r1047_merged` (16 shards)
- Waiter failed: wrong LEAN path `…-ep4-midlr/…` (real dir is `…-ep4-hilr/…`)
- **p4192:** fixed path; launched lean chall+n80 pid**113895** / vllm chall pid**114126** port**8003**

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
