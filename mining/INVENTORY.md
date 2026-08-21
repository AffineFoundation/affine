# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1133+34+R1129 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1123+R1138 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1137+R1125 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1135+R1136 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T14:56Z** | TK · **R1126+R1127 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1120+28 TRAIN** · **R1121 N80 re-arm** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1131+32 TRAIN** · **R1114 N80** |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1130 TRAIN** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1118 N80** |

Host fleet: **9 mine-*** · burn **~$392.19/h** · **wvk=7** · B200×8 stock=**0** · B300×8=**0**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T05:26:39Z | p4263: **R1121** stuck chall→re-arm r340 :8003; purge r926 11×66G (70%→1%) + r252 12×66G (34%→10%); B300/B200×8=0 |
| 2026-08-21T05:18:49Z | p4262: **R1110→R1138** UltraLoLR TRAIN r252 GPUs4,5; R1123 OK; B300/B200×8=0 |
| 2026-08-21T05:14:33Z | p4261: **R1122+24→R1135+36** r338; **R1119→R1137** r337; purge r338+r924 merges; B300/B200×8=0 |
