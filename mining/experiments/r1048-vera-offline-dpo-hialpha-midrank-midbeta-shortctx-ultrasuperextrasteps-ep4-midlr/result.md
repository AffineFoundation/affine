# R1048 — ShortCtx MidRank Midβ Ultra MidLR

**Status (p4192):** **v4 n80 LIVE** on `mine-r337` chall `:8002` GPUs6,7 after MERGE sat idle (waiter path bug).

| knob | value |
|---|---|
| base | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| β / α / r / lr | **0.1** / 128 / **32** / **1e-6** |
| ctx / steps | ShortCtx `@6144` / Ultra `28800` |
| parent | R1037 SoftCtx Midβ Mega MidLR REFUTE ~−0.79× → ShortCtx isolate |

## Timeline
- p4179: TRAIN launched GPUs6,7 pid110354 + MERGE→n80 waiter
- 2026-08-20T19:07Z train.done · 19:10Z merge.done → `/tmp/r1048_merged` (16 shards)
- Waiter failed: wrong LEAN path `…softctx-mega…-midlr/…` (real dir is `…shortctx-ultra…-midlr/…`)
- **p4192:** fixed path; launched lean chall+n80 pid**113896** / vllm chall pid**114106** port**8002**

Decision rule: Stage-5 iff margin > max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 (k=3, τ=0.03) vs reign36.
