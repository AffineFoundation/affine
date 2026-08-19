# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R912+R913 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R915+R916 TRAIN** · SSH `150.136.46.118:20299` |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R903+R904 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | TK vera TP1 · **R905+R906+R907 n80 LOADING** · SSH `18.118.83.97:40127` |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | **R910+R911 TRAIN** · SSH `95.133.252.28:40299` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · **R914 TRAIN** + R888 GRPO · SSH `192.9.163.79:20500` |

†nvidia-smi shows **7** GPUs. Host fleet: **6 mine-*** · burn **~$306.66/h** · **wvk=7**
**p4019:** brave R905+R906+R907 MERGE idle→lean_chall n80 LOADING; α→τ→Lium; B300/8×B200 stock=0

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T09:41:12Z | p4019: R905+R906+R907 MERGE→n80 LOADING; α→τ11.4→Lium; bal ~$83064; stock 0 |
| 2026-08-19T09:32:48Z | p4018: R896+R897+R898 REFUTE→R914+R915+R916 TRAIN; bal $80846; stock 0 |
| 2026-08-19T09:24:12Z | p4017: R901+R902 REFUTE→R912+R913 TRAIN; bal $80930; stock 0 |
