# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | R912+R913 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | bootstrap_r3 prewarm · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | vera-pivot teacher DL · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | vera-pivot teacher DL · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · R923 TRAIN · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | R924 MidCtx Hiβ bootstrap · SSH `31.22.104.113:40300` |

†nvidia-smi shows **7** GPUs. Host fleet: **6 mine-*** · burn **~$300.06/h** · **wvk=7**
**p4028:** rented H200 R924 (B300/non-bl B200 empty); crown R912/R913 ~105; R923 early; R337/R338 vera+teacher DL

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T18:43:46Z | p4028: rented `mine-r924` 8×H200 $33.81; burn ~$300.06/h; bal ~$79280; bl_skip=1 |
| 2026-08-19T18:37:04Z | p4027: R914 REFUTE→R923 TRAIN; R337/R338 vera-pivot; burn ~$266.26/h; bal ~$79313 |
| 2026-08-19T18:26:56Z | p4026: crown cold→R912+R913; R914 ~20/80; burn ~$266.26/h; bal ~$79377; bl_skip=1 |
