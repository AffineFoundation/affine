# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1041+R1042+R1043 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1040 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1047+R1048 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1049+R1050 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | $64.00 | **+24h from 15:50Z** | **R339 Online-DPO HiRank** · SSH `23.153.44.20:40299` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1045+R1046 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1051 TRAIN** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1044 TRAIN** · SSH `38.255.28.21:20100` |

Host fleet: **8 mine-*** · burn **~$354.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T17:53:50Z | p4180: **R1038+R1039 REFUTE** → **R1049+R1050 TRAIN** r338 pids**133218/133331**; **R1051 TRAIN** r926 pid**119797**; R1032=chal-00967; B300×8=0 |
| 2026-08-20T17:46:41Z | p4179: **R1025 REFUTE** ~0.028×; **R1037 REFUTE** ~−0.79× → **R1047+R1048 TRAIN** r337; R1032=chal-00967; r338 idle; B300×8=0 |
| 2026-08-20T17:34:04Z | p4178: **R1045+R1046 TRAIN** r924 GPUs6,7/4,5 pids**98194/98084**; R1025~50/80; R1032 not queued yet; B300×8=0 |
