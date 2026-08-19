# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | NEW · bootstrap · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | NEW · bootstrap · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | Online-DPO HiLR · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | Online-DPO BigG×HiLR · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · R888 GRPO · R914 MERGE · SSH `192.9.163.79:20500` |

†nvidia-smi shows **7** GPUs. Host fleet: **5 mine-*** · burn **~$266.26/h** · **wvk=7**
**p4024:** TTL-collapse → rented B300 crown + 3×B200; tore bl `8f34559f` R339; waiters rearmed

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T18:17:23Z | p4024: rented crown+R252+R337+R338; tore bl R339; burn ~$266.26/h; bal ~$79442 |
| 2026-08-19T10:36:28Z | p4022: R903–7 REFUTE→R918–21 TRAIN; bal ~$82620; stock 0 |
| 2026-08-19T10:22:36Z | p4021: golden R903+R904 n80 LIVE; R910 REFUTE→R917 TRAIN; bal ~$82782; stock 0 |
