# R683 — Short MidRank HiBeta UltraExtra ep3 × LoLR (offline DPO)

**Status:** **SUBMITTED** (2026-08-17 p3740) — clears live crown bar + HF pushed + on-chain.

- HF: `unconst/Affine-5czsc2fc98-r683-r252-odpo-midrank-hibeta-shortctx-ultraextra-ep3-lolr-merged@f3314c7cdc174cc4b0331bc6125c21e9b45ddbe1`
- hotkey `r683` = `5DRydxjU1Vr6ANNjnKNJvHdbmABLcVNhgojJRHDbgBRskccm`
- register extrinsic **8867097-0018** · burn ≈τ1.53 (free 1261.907→1260.382)
- `submit.py --check` OK
- commit-reveal submitted blockhash `0xf974ef15c9ed806aa50f038c1734fa0fa3f9f7428383295317b2fe56a4024322` · reveal round **31399588**
- payload: `affine1|unconst/Affine-5czsc2fc98-r683-r252-odpo-midrank-hibeta-shortctx-ultraextra-ep3-lolr-merged|f3314c7cdc174cc4b0331bc6125c21e9b45ddbe1|5DRydxjU1Vr6ANNjnKNJvHdbmABLcVNhgojJRHDbgBRskccm`
- n80 vs reign34: m=+0.002137 ~1.07× bar · thought✓(172) B✓(0.304)

## n80 vs live king reign34 (p3738)

| metric | value |
|---|---|
| king | `cryptoDev23/Affine-5Dku3dYp9j-hk8161` @ `55b7ffe0…` |
| margin | **+0.002137** |
| SE | 0.000943 |
| z | **2.27** |
| n | 79 |
| bar max(2·SE, δ) | **0.002** |
| margin/bar | **~1.07×** |
| thought median | **172** ✓ (≥80) |
| B pass | **0.304** ✓ (≥0.30) |
| wins | **true** |

## Timeline
- **p3705:** TRAIN brave 6,7 → MERGE
- **p3723:** SCP→golden 4,5/:8003 after R680 uplink clear
- **p3738:** WIN v4 ~1.07× (thought✓172 B✓0.304 knife-edge)
- **p3739:** HF card+push DONE @f3314c7c…
- **p3740:** register r683 → `--check` → **SUBMITTED**

Artifacts: `results/p3740_*.{json,txt}`, `r683_decision_reign34_wvk7.json` · keep `/tmp/r683_merged` on golden until intake.
