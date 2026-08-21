# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1160+61+62 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1167+R1168 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1171+R1172 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1164+R1165 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T15:50Z** | TK · **R1163+R1166 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1156+R1142+R1144 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1169 TRAIN** +R1155 n80 +R1159 |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1170 TRAIN** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1140 n80** |

Host fleet: **9 mine-*** · burn **~$392.19/h** · **wvk=7** · B300×8=0 · B200×8=BL-only · **R1158** waiters

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T09:42:56Z | p4291: **R1154/57/49/50 REFUTE** → **R1169+70+71+72 TRAIN**; R1155+R1140 n80 LIVE; burn **~$392.19/h** |
| 2026-08-21T09:31:43Z | p4290: fixed wait EXP paths → **R1149+R1150** n80 on r337 + **R1140** n80 on r938; R1154~71/80; burn **~$392.19/h** |
| 2026-08-21T09:23:37Z | p4289: **R1154** MERGE→Triton→probe→**n80 LIVE** pid**169229** :8003; sole B200=BL **fbb1135f**; burn **~$392.19/h** |
