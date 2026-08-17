# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-17T19:41Z** | **R664** 2,3 · **R663** 0,1 · R654/R656/R657 MERGE · **R658** · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-17T19:41Z** | TK · R537 :8002 · **R651 wait** 4,5 · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-17T19:41Z** | TK · **R647 SCP**~51G · R637 :8004 · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · **R634 REAPED** 6,7 free · **R662 TRAIN** 4,5 · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-17T19:41Z** | **R653/R659/R660/R661 TRAIN** · **R647 uplink** · R652 MERGE · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · R596 :8002 · **R655** MERGE · 6,7 free · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R337**; B300×8 **empty**; burn **~$331.45/h** · **wvk=7**
**p3681:** R634 REFUTE ~0.08×; R664 TRAIN crown 2,3; R647~51G; bal **~$89621**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T15:24:45Z | p3681: **R634 REFUTE** v4 m=+0.000497~0.08×; chall reaped; **R664 TRAIN** crown 2,3 pid**13901**; B300 empty; bal **~$89621** |
| 2026-08-17T15:18:40Z | p3680: **R663 TRAIN** crown 0,1; R634 n80 27/80; R647~26G/7sh; B300 empty; bal **~$89661** |
| 2026-08-17T15:14:14Z | p3679: **R634 SCP_READY**→lean :8003; **R647 SCP** armed; repair PID killed; B300 empty; bal **~$89702** |
