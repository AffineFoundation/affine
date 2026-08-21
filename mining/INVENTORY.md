# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-22T13:30Z** | TK · **R1215/16/17 n80** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-22T13:30Z** | TK · **R1207 MERGE+R1232 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-22T13:30Z** | TK · **R1233+R1234 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-22T13:30Z** | TK · **R1225+R1226 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | $64.00 | **2026-08-22T13:30Z** | TK · **R1227+R1228 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-22T13:30Z** | TK · **R1229+30+31 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-22T13:30Z** | TK · **R1222+23+24 TRAIN** |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-22T13:30Z** | TK · **R1220 TRAIN** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-22T13:30Z** | TK · **R1219 TRAIN** |
| mine-r1158-vera-reason-grpo-1 | eager-matrix-57 | 8×H200 | $32.00 | **2026-08-22T13:30Z** | teacher + **R1158 GRPO** |
| mine-r1191-vera-fullft-1 | swift-comet-4d | 8×H200 | $32.00 | **2026-08-22T13:30Z** | **R1221 TRAIN** |
| mine-r1214-vera-softctx-midrank-lobeta-mega-1 | brave-shark-4d | 8×H200 | $24.80 | **2026-08-22T14:43Z** | **R1218 TRAIN** GPU2 |

Host fleet: **12 mine-*** · burn **~$480.99/h** · **wvk=7** · B300/B200/H200×8 empty

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T16:50:49Z | p4338: **R1208/10/11 REFUTE** → chall reap → **R1232+33+34 TRAIN**; crown n80 ~49/27/24; burn **~$480.99/h** |
| 2026-08-21T16:44:50Z | p4337: r340 GPUs6,7 free → **R1231 TRAIN** (R1204~0.40×→Mega HiLR); crown n80 ~34/2; burn **~$480.99/h** |
| 2026-08-21T16:40:56Z | p4336: crown **R1215+16+17** MERGE idle → chall + **v4 n80 LIVE**; burn **~$480.99/h** |
