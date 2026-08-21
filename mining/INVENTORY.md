# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-22T13:30Z** | TK · orphan R1188/89/96 REFUTE |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-22T13:30Z** | TK · **R1207+R1208 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-22T13:30Z** | TK · **R1192+R1193 REFUTE** orphans |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-22T13:30Z** | TK · **R1194+R1195 REFUTE** orphans |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | $64.00 | **2026-08-22T13:30Z** | TK · **R1205+R1206 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-22T13:30Z** | TK · **R1202+R1203+R1204 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-22T13:30Z** | TK · **R1198+R1199+R1200 TRAIN** |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-22T13:30Z** | TK · **R1187 chall LOAD** :8002 |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-22T13:30Z** | TK · **R1197 REFUTE** orphan |
| mine-r1158-vera-reason-grpo-1 | eager-matrix-57 | 8×H200 | $32.00 | **2026-08-22T13:30Z** | teacher + **R1158 GRPO** |
| mine-r1191-vera-fullft-1 | swift-comet-4d | 8×H200 | $32.00 | **2026-08-22T13:30Z** | **R1209 FullFT UltraLoLR TRAIN** |

Host fleet: **11 mine-*** · burn **~$456.19/h** · **wvk=7** · B300/B200×8 empty · H200 only 4×/1× · waiters armed

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T14:26:53Z | p4322: R1187 merge→lean chall :8002 (midctx typo); stamped R1192–97 REFUTE; stock empty; burn **~$456.19/h** |
| 2026-08-21T14:20:45Z | p4321: R1201 REFUTE~−0.46×→R1209 TRAIN; R1184/85/86/1190 reaped→R1205–08 Mega TRAIN; stock empty; burn **~$456.19/h** |
| 2026-08-21T14:11:14Z | p4320: **R1201** teacher OOM@0.90 → serve util**0.85/0.80** → n80 pid**48168**; stock empty; burn **~$456.19/h** |
