# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-22T13:30Z** | TK · **R1188+R1189+R1196 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-22T13:30Z** | TK · **R1190+R1186 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-22T13:30Z** | TK · **R1192+R1193 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-22T13:30Z** | TK · **R1194+R1195 TRAIN** |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | 8×B200 | $64.00 | **2026-08-22T13:30Z** | TK · **R1184+R1185 TRAIN** |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | 8×B200 | $37.60 | **2026-08-21T21:24Z** | TK · **R1181+82+83 MERGE+wait** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-22T13:30Z** | TK · **R1198+R1199+R1200 TRAIN** |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-22T13:30Z** | TK · **R1187 TRAIN** |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-22T13:30Z** | TK · **R1197 TRAIN** |
| mine-r1158-vera-reason-grpo-1 | eager-matrix-57 | 8×H200 | $32.00 | **2026-08-22T11:23Z** | teacher + **R1158 GRPO** |
| mine-r1191-vera-fullft-1 | swift-comet-4d | 8×H200 | $32.00 | **2026-08-22T11:53Z** | **R1191 TKC→n80** |

Host fleet: **11 mine-*** · burn **~$456.19/h** · **wvk=7** · B300/H200×8 empty · B200=`fbb1135f` BL · waiters armed

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-chat`, `swarm-t-h200-4x-1`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-21T12:47:37Z | p4314: **R1191** HF-push abort → **local TKC** `SKIP_LOCAL_TKC=0` pid**6151** (T+K loading); stock BL-only B200; burn **~$456.19/h** |
| 2026-08-21T12:42:24Z | p4313: **R1178 REFUTE ~−0.58×** reaped r924 :8002 → **R1200 TRAIN** pid**190717** GPUs6,7; R1198/99 OK; stock empty; burn **~$456.19/h** |
| 2026-08-21T12:33:28Z | p4312: **R1169+R1173 REFUTE** reaped r924 → **R1198+R1199 TRAIN** (pids **189113/189151**); R1178 n80 ~52/80; BL-only B200; burn **~$456.19/h** |
