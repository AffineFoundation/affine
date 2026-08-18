# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R758+R763 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R765+R764 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R755+R756 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R757+R766 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R761–R768 TRAIN** SoftCtx Mega |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R759 N80** + **R760 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `lium exec gentle-wolf-8c` (direct :40299 flaky)
Host fleet: **6 mine-*** · B300×8 **0** · burn **~$331.45/h** · **wvk=7**
**p3816:** R759 N80 LIVE chall/:8002 sim357410; waiters HEAD=R337; bal **~$86639.43**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T06:40:53Z | p3816: **R759 N80 LIVE** R252 4,5/:8002; B300×8=0; bal **~$86639.43** |
| 2026-08-18T06:35:20Z | p3815: **API BL fix** fbb1135f comment-break → restart api **2852754**; B300×8=0; bal **~$86721.17** |
| 2026-08-18T06:29:54Z | p3814: **brave idle 4–7→R767+R768 TRAIN** SoftCtx LoBeta Mega; B300×8=0; bal **~$86762.78** |
