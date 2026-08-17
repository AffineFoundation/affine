# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R705 TRAIN** 4,5 · R698 MERGE 6,7 · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R701 TRAIN** 4,5 · keep `/tmp/r675_merged` · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R683 HF DONE** @`f3314c7c…` · R637 :8004 · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R703 TRAIN** 4,5 · **R702 TRAIN** 6,7 · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | R699 0,1 · R700 2,3 · R696 4,5 · **R704 TRAIN** 6,7 · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R706 TRAIN** 4,5 · **R707 TRAIN** 6,7 · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R683 HF→submit**; B300×8 **empty**; burn **~$331.45/h** · **wvk=7**
**p3739:** **R683 HF DONE** `…r683-…-merged`@`f3314c7c…` (~70G); bal **~$86722**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T21:21:46Z | p3739: **R683 HF DONE** @`f3314c7c…` (~70G; token refresh); B300 empty; bal **~$86722** |
| 2026-08-17T21:17:48Z | p3738: **R707 TRAIN** R252 6,7 + **R683 WIN** harvest (~1.07×); B300 empty; bal **~$86763** |
| 2026-08-17T21:13:08Z | p3737: **R705+R706 TRAIN** fill crown/R252 idle 4,5; R683 N80~44/80; B300 empty; bal **~$86804** |
