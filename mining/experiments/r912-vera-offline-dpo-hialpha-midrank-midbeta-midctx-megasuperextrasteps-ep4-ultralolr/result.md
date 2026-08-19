# R912 result — p4034

## Verdict: **REFUTE** (Reason v4 / wvk=7)

| field | value |
|---|---|
| king | reign36 `vera6/affine-5g4yy75zuz-t6`@`8e3f1695` |
| margin | **+0.005202** |
| SE | 0.004877 |
| bar = max(2·SE, δ) | **0.009753** |
| ratio | **~0.53×** |
| z | 1.067 |
| n | 79 |
| thought median | 167 ✓ (≥80) |
| B pass | 0.506 ✓ (≥0.30) |
| duel_params | k=3 · τ=0.03 |

## Axis
vera Offline-DPO Soft Mid Mid Soft · HiAlpha · MidRank r=32 · MidBeta β=0.1 · MidCtx @8192 · ep4 · UltraLoLR 5e-7 · max_steps=19200

## Ops note (p4034)
Cold crown bootstrap omitted `s4-h2-merge/run_sim_duel.py` and pyarrow; n80 first launch failed. Fixed: uploaded sim, `ensurepip`+pyarrow, corpus sync epoch=13, relaunched. Challenger stayed warm through fix.

## Follow-up
Chall :8003 reaped → **R928** HiRank r=64 Midβ MidCtx TRAIN on GPUs 6,7.
