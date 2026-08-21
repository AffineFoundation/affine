# R1187 — cryptoDev SoftCtx MidRank MidLoβ MidLR (p4304)

Parent **R1170** MidCtx MidRank MidLoβ Hyper MidLR REFUTE vs reign36 wvk7:
- m=**+0.004094** SE=0.003599 z=1.138 n=80 bar≈0.007197 (~**0.57×**)
- thought✓200 B✓0.399 k=3 τ=0.03

Axis: cryptoDev23 Offline-DPO HiAlpha MidRank MidLoβ **SoftCtx** HyperSuperExtra ep4 **MidLR** (β=0.05 r=32 α=128 lr=**1e-6** @**12288** max_steps=38400)
≠ MidCtx MidLR R1170 / ≠ MidCtx UltraLoLR R1157 / ≠ MidCtx HiLR R1130 / ≠ SoftCtx Mega UltraLoLR R926 / ≠ Online / ≠ GRPO

## Decision
Stage-5 iff fresh v4 n80 margin>max(2·SE, δ=0.002) AND thought≥80 AND B≥0.30 vs reign36.

## p4322 — merge→n80 unstuck
- Train DONE; merge DONE **2026-08-21T13:37:57Z** (`/tmp/r1187_merged`, 16 shards, weight_identical=false).
- Waiter failed: LEAN path typo `…midctx…` vs SoftCtx EXP dir → chmod miss → exit; GPUs 3,4 idle ~48m.
- Patched wait path softctx; relaunched `lean_chall_n80_r926_gpus34_p4304.sh` pid**180517**; chall vLLM **:8002** TP2 util0.72 pid**180573**.
