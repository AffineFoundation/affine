# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | R912+R913 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | R3 GRPO TRAIN · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | vera online-DPO · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | vera online-DPO BigG · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · R923 TRAIN · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | R924+R925 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | R926 SoftCtx MidLoβ bootstrap · SSH `93.120.231.186:32301` |

†nvidia-smi shows **7** GPUs. Host fleet: **7 mine-*** · burn **~$313.82/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T19:22:56Z | p4030: R925 TRAIN on R924 GPUs2,3; rented `mine-r926` 8×H100 $13.76 SoftCtx MidLoβ bootstrap; burn ~$313.82/h; bal ~$79000 |
| 2026-08-19T18:43:46Z | p4028: rented `mine-r924` 8×H200 $33.81; burn ~$300.06/h; bal ~$79280; bl_skip=1 |
| 2026-08-19T18:37:04Z | p4027: R914 REFUTE→R923 TRAIN; R337/R338 vera-pivot; burn ~$266.26/h; bal ~$79313 |
