# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1146+47+48 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1138+R1153 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1149+R1150 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1151+R1152 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T15:50Z** | TK · **R1143+R1145 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1156+R1142+R1144 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1159 TRAIN** +R1154+55 |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1157 TRAIN** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1140 TRAIN** |
| mine-r1158-vera-reason-grpo-1 | brave-matrix-2a | 8×B200? | $5.60 | **2026-08-22T07:45Z** | BOOT GRPO · **visible 1×B200** |

Host fleet: **10 mine-*** · burn **~$397.79/h** · **wvk=7** · B300×8=0 · B200×8=0

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T08:00:48Z | p4280: **R1139 REFUTE** ~−0.26× → exact-PID reap :8002 → **R1159** UltraLoLR TRAIN pid**164539** GPUs6,7; **r1158** stack+pip+king DL ✓ teacher DL; burn **~$397.79/h** |
| 2026-08-21T07:47:51Z | p4279: **R1130 REFUTE** ~−0.40× → **R1157** TRAIN; **R1139** n80 armed; **rent** `mine-r1158` 8×B200 $5.60/h |
| 2026-08-21T07:32:55Z | p4278: r340 **R1141 REFUTE** ~−0.43× → **R1156** Midβ UltraLoLR TRAIN; r926 **R1130** ~40/80; B300/B200×8=0 |
