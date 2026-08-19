# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R885+R886 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R864 relay→n80** · **R874 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R877+R878 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | TK vera TP1 · **R875+R876+R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | TK **vera** · **R872+R873 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | teacher · **R879–R881 TRAIN** · R869–R871 MERGE |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R882–R884 TRAIN** · R864 relay · R865 MERGE |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: **host:40299 TIMEOUT** — use `lium exec gentle-wolf-8c` / `lium scp`
SSH R337: `ssh root@86.38.182.67 -p 20295`
SSH R338: `ssh root@86.38.182.55 -p 20299`
Host fleet: **7 mine-*** · B300×8=0 · B200×8 listed · burn **~$366.50/h** · **wvk=7**
**p3977:** R863 REFUTE→R864 host-relay LIVE; brave idle 2,3→R887 MidLoβ TRAIN; stock B300=0

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T03:30:31Z | p3977: R863 REFUTE→reap→R864 relay; brave idle→R887 TRAIN; stock 0 |
| 2026-08-19T03:22:16Z | p3976: crown R861/R862 reap→R885/R886 TRAIN; R863 n80 LIVE; stock 0 |
| 2026-08-19T03:14:19Z | p3975: R337/R338 idle 2–7→R879–R884 TRAIN; R863~63G; stock 0 |
