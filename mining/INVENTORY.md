# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1091+92+93** · SSH OK |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1090+R1081** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1083+R1094** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1087+R1088** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T14:56Z** | TK · **R1085+R1086** · SSH `23.153.44.20:40299` |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | **R340+R1096+R1097** · king GPU5 · SSH `18.118.83.97:40127` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1084+R1089+R1095** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1080** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1082** · SSH `38.255.28.21:20100` |

Host fleet: **9 mine-*** · burn **~$392.18/h** · **wvk=7** · B200×8 stock=**0** · B300×8=**0**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T00:09:45Z | p4226: crown SSH OK; r340 idle GPUs1–4 → **R1096+R1097 TRAIN** + king GPU5; B300/B200×8=0 |
| 2026-08-21T00:03:01Z | p4225: r924 R1068 idle :8004 reap → **R1095 TRAIN** pid**125347** GPUs4,5; crown SSH timeout; B300/B200×8=0 |
| 2026-08-20T23:56:33Z | p4224: r337 R1064 idle :8003 reap → **R1094 TRAIN** pid**135918** GPUs4,5; B300/B200×8=0 |
