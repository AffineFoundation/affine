# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R1019** 1,3 + **R1021+R1022 TRAIN** 6,7+4,5 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | TK · **R1020 TRAIN** 6,7 + R1014 TRAIN 4,5 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | TK · **R1023+R1024 TRAIN** 6,7+4,5 · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | TK · **R1018+R1017 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R1016+R1015+R1005 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | T+K · **R1013 TRAIN** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-20T21:23Z** | TK · **R1012 TRAIN** · SSH `38.255.28.21:20100` |

Host fleet: **7 mine-*** · burn **~$290.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T13:18:16Z | p4156: **R1004+R1011 REFUTE**→**R1023+R1024 TRAIN** r337 6,7+4,5 pids92389/92386; B300×8=0 |
| 2026-08-20T13:11:52Z | p4155: **R1009+R1010 REFUTE**→**R1021+R1022 TRAIN** crown 6,7+4,5; B300×8=0 |
| 2026-08-20T13:05:59Z | p4154: **R998 REFUTE**→**R1020 TRAIN** r252 6,7 pid119314 + MERGE→n80; B300×8=0 |
