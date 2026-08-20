# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-21T13:26Z** | TK · **R1066+R1067+R1069 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-21T13:26Z** | TK · **R1055+R1058 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-21T13:26Z** | TK · **R1064+R1063 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-21T13:26Z** | TK · **R1061+R1065 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | soft **2026-08-21T14:56Z** | TK · **R1070+R1071 TRAIN** · SSH `23.153.44.20:40299` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-21T13:26Z** | TK · **R1045+R1054+R1068 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-21T13:26Z** | TK · **R1051+R1060 TRAIN** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-21T13:26Z** | TK · **R1062 TRAIN** · SSH `38.255.28.21:20100` |

Host fleet: **8 mine-*** · burn **~$354.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T20:26:42Z | p4201: **R1052 REFUTE** m=−0.001866 ~−0.40× → reap r339 :8003 → **R1071 TRAIN** pid**28262**; R1070 ok; B300×8=0 B200×8=0 |
| 2026-08-20T20:20:42Z | p4200: **R1052 Triton-reseed** orphan EngineCore reap → seed `chall_r1053` (n_so=26) → CHALL_READY + **n80 LIVE** pid**27596**; R1070 ok; B300×8=0 |
| 2026-08-20T20:10:54Z | p4199: **R1059 REFUTE** → reap crown :8003 → **R1069 TRAIN** pid**240048**; **R1053 REFUTE** → reap r339 :8002 → **R1070 TRAIN** pid**24092**; B300×8=0 |
