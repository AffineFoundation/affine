# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1103+04+01 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1110+R1105 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1106+R1107 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1111+R1102 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T14:56Z** | TK · **R1108+R1109 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | **R340+R1096+R1097** · king:8001 |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1112 TRAIN** · R1089 n80 |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1098** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1099 TRAIN** |

Host fleet: **9 mine-*** · burn **~$392.18/h** · **wvk=7** · B200×8 stock=**0** · B300×8=**0**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T02:27:50Z | p4241: **R1095 REFUTE** → reap r924 :8004 → **R1112 TRAIN** pid**136444** (R1089 n80 intact); B300/B200×8=0 |
| 2026-08-21T02:22:27Z | p4240: **R1100 REFUTE** → reap r338 :8003 → **R1111 TRAIN** pid**182821** (R1102 intact); B300/B200×8=0 |
| 2026-08-21T02:17:46Z | p4239: **R1090 REFUTE** → reap r252 :8002 → **R1110 TRAIN** pid**180255**; B300/B200×8=0; note **R1100 REFUTE** idle r338 |
