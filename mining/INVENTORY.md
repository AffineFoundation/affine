# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R981+R982 v4 n80** :8002/3 · **R993 TRAIN** 1,3 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | TK · **R988+R975 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | TK · R987+R989 TRAIN · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | TK · **R991+R990 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R986+R984+R985 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | T+K · **R973 v4 n80** :8002 · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-20T21:23Z** | TK · **R992 TRAIN** 2,3 · SSH `38.255.28.21:20100` |

Host fleet: **7 mine-*** · burn **~$290.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T09:28:20Z | p4123: **R983 REFUTE**→exact-PID reap→**R993 TRAIN** crown 1,3; R981+R982 ~79/80; R973 ~30/80; B300×8=0 |
| 2026-08-20T09:24:20Z | p4122: **R980 REFUTE**→exact-PID reap→**R992 TRAIN** R938; crown n80 mid; R973 ~9/80; B300×8=0 |
| 2026-08-20T09:19:36Z | p4121: R973 HF-layout serve→**v4 n80 LIVE** R926 :8002; crown n80 mid; B300×8=0 |
