# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R994+R995+R993 TRAIN** 6,7/4,5/1,3 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | TK · **R998+R1002 TRAIN** 6,7/4,5 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | TK · **R989 n80** :8002 6,7 + **R997 TRAIN** 4,5 · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | TK · **R999+R1000 TRAIN** 6,7/4,5 · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R1001+R986+R1003 TRAIN** + **R986 MERGE→n80 waiter** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | T+K · **R996 TRAIN** 3,4 · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-20T21:23Z** | TK · **R992 TRAIN** 2,3 · SSH `38.255.28.21:20100` |

Host fleet: **7 mine-*** · burn **~$290.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T10:31:38Z | p4132: **R989 MERGE→v4 n80** R337 :8002 GPUs6,7 pid79352; R959 scoring 81/1300; B300×8=0 BL-only |
| 2026-08-20T10:24:17Z | p4131: **R986 MERGE→n80 waiter** armed R924; R973 REFUTE confirmed (→R996 already); B300×8=0 BL-only |
| 2026-08-20T10:18:13Z | p4130: **R985 REFUTE**→reap→**R1003 TRAIN** R924 4,5; B300×8=0 (1×B300 only) |
