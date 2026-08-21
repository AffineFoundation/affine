# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1160+R1161 TRAIN** +R1147 n80 |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1138+R1153 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1149+R1150 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1151+R1152 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T15:50Z** | TK · **R1143+R1145 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1156+R1142+R1144 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1159 TRAIN** +R1154+55 |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1157 TRAIN** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1140 TRAIN** |

Host fleet: **9 mine-*** · burn **~$392.19/h** · **wvk=7** · B300×8=0 · B200×8 stock=BL-only · **R1158 slot open** (waiters)

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T08:26:00Z | p4282: **R1146+R1148 REFUTE** → reap chall :8002/:8003 → **R1160+R1161** TRAIN; R1147 n80 :8004; burn **~$392.19/h** |
| 2026-08-21T08:10:30Z | p4281: tore **brave-matrix-2a** + **eager-lion-45** (1/8 GPU, exec **fbb1135f**); BL note; R1158 waiter=node-id+ngpu≥8 |
| 2026-08-21T08:00:48Z | p4280: **R1139 REFUTE** ~−0.26× → **R1159** UltraLoLR TRAIN; **r1158** bootstrap LIVE (1/8 GPU) |
