# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · R1066/67/69 REFUTE · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1081+R1076 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1078 REFUTE** slot free · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1072+R1077** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T14:56Z** | TK · R1070/71 REFUTE · SSH `23.153.44.20:40299` |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | **R340 TRAIN** · SSH `18.118.83.97:40127` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1074 REFUTE** + R1073 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1080 TRAIN** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1079 n80 LIVE** · SSH `38.255.28.21:20100` |

Host fleet: **9 mine-*** · burn **~$392.18/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T23:03:35Z | p4215: **R1075 REFUTE** ~0.53× → **R1081 TRAIN** pid162736 GPUs6,7; R1079 n80~66/80; R340≥10; B300×8=0 |
| 2026-08-20T22:56:47Z | p4214: **R1051 REFUTE** ~0.27× → **R1080 TRAIN** pid145066 GPUs3,4; R340 step≥6; R1064 Qpos3; B300×8=0 |
| 2026-08-20T22:39:38Z | p4213: **R1060 REFUTE**; **R340** train+post waiter; R1051 n80 ~5/80; R1064 queued; B300×8=0 |
