# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R885+R886 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R865 relay** · **R874 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R877+R878 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | TK vera TP1 · **R875+R876 n80** · **R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | TK **vera** · **R872+R873 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | teacher · **R879–R881 TRAIN** · R871 src |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R882–R884 TRAIN** · R865 src |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king READY · R888 GRPO · R871 relay · SSH `192.9.163.79:20500` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$405.70/h** · **wvk=7**
**p3984:** R875+R876 MERGE_DONE→n80 LIVE brave :8002/:8003 TP1; R865~14/16; R871~4/16; R888 king READY GRPO≥27; B300/8×B200 stock=0

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T04:21:57Z | p3984: R875/R876 MERGE_DONE→n80 LIVE brave :8002/:8003 TP1; R888 king READY; burn ~$405.70/h; bal $83890 |
| 2026-08-19T04:12:52Z | p3983: R888 idle→king TP1+R871 relay; R865~33G; R888 GRPO ok; burn ~$405.70/h; bal $83939 |
| 2026-08-19T04:03:52Z | p3982: R864 REFUTE m=−0.01080~−1.12×; reap lunar 4,5; R865 relay LIVE; R888 GRPO TRAIN; burn ~$405.70/h; bal $84033 |
