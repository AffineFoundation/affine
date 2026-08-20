# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R994+R995+R993 TRAIN** 6,7/4,5/1,3 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | TK · **R998+R1002 TRAIN** 6,7/4,5 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | TK · **R1004+R997 TRAIN** + **MERGE→n80** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | TK · **R999+R1000 TRAIN** 6,7/4,5 · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R1005+R1001+R1003 TRAIN** + **MERGE→n80** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | T+K · **R996 TRAIN** 3,4 · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-20T21:23Z** | TK · **R992 TRAIN** 2,3 · SSH `38.255.28.21:20100` |

Host fleet: **7 mine-*** · burn **~$290.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T10:56:13Z | p4136: **R986 REFUTE**→reap→**R1005 TRAIN** R924 6,7; R959 760/1300; B300×8=0 |
| 2026-08-20T10:48:13Z | p4135: **R1001+R1003 MERGE→n80** armed R924; R986 ~31/80; R959 535/1300; B300×8=0 |
| 2026-08-20T10:44:05Z | p4134: **R997 MERGE→n80** armed R337; R986 n80 LIVE; R959 369/1300; B300×8=0 |
