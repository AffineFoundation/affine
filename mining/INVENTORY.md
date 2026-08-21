# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1146+47+48 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1138+R1153 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1149+R1150 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1151+R1152 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T14:56Z** | TK · **R1143+R1145 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1142+R1141+R1144 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1132 n80 + R1131 chall + R1139 TRAIN** |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1130 n80 FATAL — re-arm** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1140 TRAIN** |

Host fleet: **9 mine-*** · burn **~$392.19/h** · **wvk=7** · B200×8 stock=**0** · B300×8=**0**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T07:05:50Z | p4274: r252 **R1123 REFUTE** ~−0.11× → exact-PID reap :8003 → **R1153** UltraLoLR TRAIN pid**198900** GPUs6,7; R1130 n80 FATAL noted; B300/B200×8=0 |
| 2026-08-21T07:00:17Z | p4273: r337 **R1125+R1137 REFUTE**→reap :8002/:8003 → **R1149+R1150** TRAIN pids**166378/166373**; r338 **R1135+R1136 REFUTE**→reap → **R1151+R1152** TRAIN pids**212204/212198**; B300/B200×8=0 |
| 2026-08-21T06:52:18Z | p4272: crown **R1129+R1133+R1134 REFUTE**→exact-PID reap :8002/:8004/:8003 → **R1146+R1147+R1148** TRAIN pids**305741/305738/305731**; B300/B200×8=0 |
