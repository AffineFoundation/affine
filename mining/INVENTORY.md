# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R1006+R1009+R1010 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | TK · **R998 TRAIN** 6,7 · **R1014 TRAIN** 4,5 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | TK · **R1004+R1011 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | TK · **R1007+R1008 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R1015+R1003+R1005 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | T+K · **R1013 TRAIN** GPUs3,4 · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-20T21:23Z** | TK · **R1012 TRAIN** GPUs2,3 · SSH `38.255.28.21:20100` |

Host fleet: **7 mine-*** · burn **~$290.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T12:19:47Z | p4149: **R1001 REFUTE**→**R1015 MidLR TRAIN** r924 1,3 pid70313; B300×8=0 BL-only `8f34559f` |
| 2026-08-20T12:05:36Z | p4148: **R996 REFUTE**→**R1013 MidLR TRAIN** r926; **R1002 REFUTE**→**R1014 MidLR TRAIN** r252; B300×8=0 |
| 2026-08-20T11:56:37Z | p4147: rented `fbb1135f` (already BL; 8→3 GPU lie) → **rm** brave-fox-5c; R996~58/80; R1002 chall loading; B300×8=0 |
