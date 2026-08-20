# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1019+R1021+R1022 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1020+R1014 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1023+R1024 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1018+R1017 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | **R1016+R1015+R1005 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | T+K · **R1013+R1025 TRAIN** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1012 TRAIN** · SSH `38.255.28.21:20100` |

Host fleet: **7 mine-*** · burn **~$290.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T13:41:35Z | p4159: r926 teacher TP4→TP2 → **R1025 TRAIN** GPUs5,6; B300×8=0; BL B200 only; R1008 q#1 |
| 2026-08-20T13:26:53Z | p4158: TTL+Soft/Dead →**2026-08-21T13:26Z** all 7 mine-*; B300×8=0; BL B200 only; R1008 q#1 |
| 2026-08-20T13:23:11Z | p4157: α→τ→Lium r252 swept; B300×8=0; B200 catalog=BL-only; fleet 7 TRAIN ok |
