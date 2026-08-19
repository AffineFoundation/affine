# R861 — SoftCtx × MidRank × MidBeta UltraLoLR (offline DPO on vera)

**Status:** **SUBMITTED** (2026-08-19 p3967) — clears live crown bar + HF pushed + on-chain.

- HF: `unconst/Affine-5czsc2fc98-r861-vera-odpo-midrank-midbeta-softctx-megaextra-ep4-ultralolr-merged@f20753584e79f874e3c1984211a8653bc8785fc2`
- hotkey `r861` = `5FBwqMmqnj2uGFK3UU8XWTxaf9M5VVH5v6YniKAeisKJ69aA`
- register extrinsic **8875721-0018** · burn ≈τ2.83 (free 1260.446→1257.618)
- `submit.py --check` OK
- commit-reveal submitted blockhash `0x8f9a17fc26ed55e3…` · reveal round **31434087**
- payload: `affine1|unconst/Affine-5czsc2fc98-r861-vera-odpo-midrank-midbeta-softctx-megaextra-ep4-ultralolr-merged|f20753584e79f874e3c1984211a8653bc8785fc2|5FBwqMmqnj2uGFK3UU8XWTxaf9M5VVH5v6YniKAeisKJ69aA`
- n80 vs reign36: m=+0.003665 ~1.088× bar · thought✓(141.5) B✓(0.5375)

## n80 vs live king reign36 (p3966)

| metric | value |
|---|---|
| king | `vera6/affine-5g4yy75zuz-t6` @ `8e3f1695…` |
| margin | **+0.003665** |
| SE | 0.001684 |
| z | **2.177** |
| n | 80 |
| bar max(2·SE, δ) | **0.003367** |
| margin/bar | **~1.088×** |
| thought median | **141.5** ✓ (≥80) |
| B pass | **0.5375** ✓ (≥0.30) |
| k / τ | **3** / **0.03** |

## Timeline
- **p3951:** TRAIN after R846 REFUTE
- **p3966:** CLEAR → HF push LIVE (failed: HF public storage quota)
- **p3967:** purge LOST merges → HF push OK `@f2075358…` → register → `--check` → **SUBMITTED**

Artifacts: `results/p3967_*.{json,txt}`, `r861_decision_reign36_wvk7.json`
