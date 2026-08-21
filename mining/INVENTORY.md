# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1188+R1189+R1179 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1190+R1186 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1171+R1172 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1177+R1180 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T15:50Z** | TK · **R1184+R1185 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1181+82+83 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1169+73+78 TRAIN** |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1187 TRAIN** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1174 TRAIN** |
| mine-r1158-vera-reason-grpo-1 | eager-matrix-57 | 8×H200 | $32.00 | **2026-08-22T11:23Z** | teacher + **R1158 GRPO TRAIN** |

Host fleet: **10 mine-*** · burn **~$424.19/h** · **wvk=7** · B300/H200 empty · sole B200=`fbb1135f` BL · waiters armed

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T11:49:14Z | p4306: **R1168 REFUTE~−0.42×** → reap :8002 → **R1190 TRAIN** r252 pid**221526**; sole B200 BL; burn **~$424.19/h** |
| 2026-08-21T11:43:58Z | p4305: **R1175 REFUTE~−0.24×** + **R1176 REFUTE~−0.29×** → **R1188+R1189 TRAIN**; R1168 n80~25/80; burn **~$424.19/h** |
| 2026-08-21T11:35:48Z | p4304: **R1158** teacher+GRPO TRAIN pid**5163**; **R1170 REFUTE~0.57×**→**R1187 TRAIN** r926; burn **~$424.19/h** |
