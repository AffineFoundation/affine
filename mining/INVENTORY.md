# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R885+R886 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R896+R897 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R877+R878 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | TK vera TP1 · **R889+R890+R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | TK **vera** · **R892+R893 n80 LIVE** · SSH `95.133.252.28:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | **R895+R894 TRAIN** 4–7 · SSH `86.38.182.67:20295` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R882–R884 TRAIN** |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · **R888 GRPO** + **R898 TRAIN** 5,6 · SSH `192.9.163.79:20500` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$405.70/h** · **wvk=7**
**p4000:** R898 TRAIN+wait→merge R888; R892+R893 n80 ~24/~25; B300/8×B200 stock=0

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T07:03:03Z | p4000: R898 TRAIN R888 5,6 + wait→merge; R892+R893 n80 LIVE; burn ~$405.70/h; bal $82303 |
| 2026-08-19T06:55:40Z | p3999: R892+R893 TRAIN_DONE→MERGE+wait n80 R252; burn ~$405.70/h; bal $82358 |
| 2026-08-19T06:49:00Z | p3998: R891+R874 REFUTE→R896+R897 TRAIN lunar; burn ~$405.70/h; bal $82553 |
