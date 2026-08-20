# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1041+R1042+R1043 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1040 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1037 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1038+R1039 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | $64.00 | **+24h from 15:50Z** | **R339 Online-DPO HiRank** · SSH `23.153.44.20:40299` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | **R1032 HF PUSH** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TKC · **R1025 n80** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1044 TRAIN** · SSH `38.255.28.21:20100` |

Host fleet: **8 mine-*** · burn **~$354.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T17:30:16Z | p4177: **R1032 SUBMITTED** HF@`62dfb322` reg**8887516-0019** reveal**31481268**; purged LOST ~261GB; R1025@14/80; B300×8=0 |
| 2026-08-20T17:20:52Z | p4176: **R1029 REFUTE→R1044 ShortCtx TRAIN** r938 pid**35545**; R1025 n80 early; B300/B200×8=0 |
| 2026-08-20T17:15:15Z | p4175: **R1025 chall OOM→rearm** util0.65+expandable :8003 UP · n80 pid**118288**; crown trains ~step70; B300/B200×8=0 |
