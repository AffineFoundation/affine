# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R885+R886 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R891+R874 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R877+R878 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | TK vera TP1 · **R889+R890+R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | TK **vera** · **R872 n80** :8002 · SSH `95.133.252.28:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | teacher · R879–R881 · R871 src |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R882–R884 TRAIN** · R865 src |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king+R871 chall :8002 · **R871 n80~78/80** · R888 GRPO · SSH `192.9.163.79:20500` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$405.70/h** · **wvk=7**
**p3988:** R872 MERGE→chall+n80 LIVE R252; R871~78/80; B300/8×B200 stock=0

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T05:00:43Z | p3988: R872 MERGE→chall :8002 + n80 LIVE pid584389; R871~78/80; burn ~$405.70/h; bal $83462 |
| 2026-08-19T04:53:27Z | p3987: R871 SIZE_OK→chall+n80 LIVE; affine_pkg sync; burn ~$405.70/h; bal $83559 |
| 2026-08-19T04:38:17Z | p3986: R875/R876/R865 REFUTE→reap+R889/R890/R891 TRAIN; burn ~$405.70/h; bal $83701 |
