# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | cold boot R912+R913 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | bootstrap_r3 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | bootstrap_h139 · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | bootstrap · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · R914 n80 ~20/80 · SSH `192.9.163.79:20500` |

†nvidia-smi shows **7** GPUs. Host fleet: **5 mine-*** · burn **~$266.26/h** · **wvk=7**
**p4026:** crown cold-boot R912+R913; R914 ~20/80; stock bl-only

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T18:26:56Z | p4026: crown cold→R912+R913; R914 ~20/80; burn ~$266.26/h; bal ~$79377; bl_skip=1 |
| 2026-08-19T18:21:36Z | p4025: R914 n80 LIVE; R252/R337/R338 bootstrap; crown empty; burn ~$266.26/h; bal ~$79409 |
| 2026-08-19T18:17:23Z | p4024: rented crown+R252+R337+R338; tore bl R339; burn ~$266.26/h; bal ~$79442 |
