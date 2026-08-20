# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1041+R1042+R1043 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1030+R1040 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1034+R1037 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1038+R1039 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | $64.00 | **+24h from 15:50Z** | **R339 Online-DPO HiRank** · SSH `23.153.44.20:40299` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | **R1032+R1026+R1033** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1025 n80 rearm** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · stale **R1029** · SSH `38.255.28.21:20100` |

Host fleet: **8 mine-*** · burn **~$354.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T17:05:15Z | p4174: **R1031+R1035+R1036 REFUTE→R1041/42/43 TRAIN** crown; R1025 chall DOWN; B300/B200×8=0 |
| 2026-08-20T15:57:20Z | p4171: **R339** bootstrap 404→**vera6** pin + relaunch pid**1868**; fleet 8; R1025 n80 ~34/80; B300/B200×8=0 |
| 2026-08-20T15:51:54Z | p4170: **R1020 REFUTE** B✗ → **R1040 TRAIN** r252 6,7 pid**132110**; R1025 n80 1/80; B300/B200×8=0 |
