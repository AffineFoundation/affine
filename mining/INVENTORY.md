# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · R861 **scoring** · R862 idle |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R866+R860 n80 LIVE** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · R858/R859 REFUTE idle |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | TK vera TP1 · **R850+R851 n80** ~65–68/80 |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | TK **vera** · **R872+R873 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | teacher · R869–R871 merged · 2–7 idle |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | R863–R865 adapters · merges stalled · 2–7 idle |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: **host:40299 TIMEOUT** — use `lium exec gentle-wolf-8c` / `lium scp`
SSH R337: `ssh root@86.38.182.67 -p 20295`
SSH R338: `ssh root@86.38.182.55 -p 20299`
Host fleet: **7 mine-*** · B300×8=0 · B200×8=bl(`fbb1135f`) · burn **~$366.49/h** · **wvk=7**
**p3970:** R867/R868 REFUTE; lunar R866+R860 n80; R252 R872/R873 TRAIN

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T02:33:40Z | p3970: R867/R868 REFUTE→R872/R873 TRAIN; lunar orphans→R866+R860 n80 LIVE; R861~1034/1300; stock bl-only |
| 2026-08-19T02:24:18Z | p3969: brave R850/R851 n80 LIVE (flashinfer fix); R867/R868 ~40/80; stock 0 |
| 2026-08-19T02:19:27Z | p3968: R252 R867/R868 CHALL_READY + n80 LIVE; stock 0 |
