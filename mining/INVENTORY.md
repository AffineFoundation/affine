# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1175+76+79 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1168 + R1186 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1171+R1172 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1177+R1180 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T15:50Z** | TK · **R1184+R1185 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1181+82+83 TRAIN** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1169+73+78 TRAIN** |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1170 n80** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1174 TRAIN** |
| mine-r1158-vera-reason-grpo-1 | eager-matrix-57 | 8×H200 | $32.00 | **2026-08-22T11:23Z** | **R1158 BOOTSTRAP** |

Host fleet: **10 mine-*** · burn **~$424.19/h** · **wvk=7** · B300×8=0 · B200×8=0 · H200×8=0 · fleet waiters still armed

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T11:24:40Z | p4303: rented **R1158** 8×H200 `eager-matrix-57`@$32 node `e350ebc9…`/`golden-orbit-7b` (ssh_gpus=8); bootstrap pid**1234**; burn **~$424.19/h** |
| 2026-08-21T11:18:06Z | p4302: **R1167 REFUTE ~−0.15×** → **R1186 TRAIN** r252 GPUs6,7 pid**216502**; burn **~$392.19/h** |
| 2026-08-21T11:06:11Z | p4301: **R1166 REFUTE ~0.45×** → **R1185 TRAIN** r339 GPUs6,7 pid**87065**; burn **~$392.19/h** |
