# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R901+R902 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R896+R897 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R903+R904 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | TK vera TP1 · **R905+R906+R907 TRAIN** · SSH `18.118.83.97:40127` |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | TK **vera** · **R899+R900 TRAIN** · SSH `95.133.252.28:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | **R895+R894 TRAIN** · SSH `86.38.182.67:20295` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R882–R884 MERGE** idle host-relay |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · **R898 TRAIN** · SSH `192.9.163.79:20500` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$405.70/h** · **wvk=7**
**p4006:** R890+R887+R889 REFUTE→R905+R906+R907 TRAIN brave; B300/8×B200 stock=0

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T07:51:26Z | p4006: R890+R887+R889 REFUTE→R905+R906+R907 TRAIN brave; burn ~$405.70/h; bal $81852 |
| 2026-08-19T07:42:04Z | p4005: R877+R878 REFUTE→R903+R904 TRAIN golden; burn ~$405.70/h; bal $81903 |
| 2026-08-19T07:33:40Z | p4004: R885+R886 REFUTE→R901+R902 TRAIN; R887 lean_chall; burn ~$405.70/h; bal $82003 |
