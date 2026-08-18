# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R758 TRAIN** · **R753 N80** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R752 TRAIN** · **R749 N80** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R755+R756 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R757+R754 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R761+R762 TRAIN** SoftCtx Mega |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R759+R760 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `lium exec gentle-wolf-8c` (direct :40299 flaky)
Host fleet: **R761+R762 TRAIN** brave idle fill; R749+R753 **N80 LIVE**; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3808:** brave idle→**R761 Soft Midβ SoftCtx Mega** 0,1 pid**276621** + **R762 Soft Hiβ SoftCtx Mega** 2,3 pid**276622** wait→merge; bal **~$87088.37**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T05:47:40Z | p3808: **brave idle→R761+R762 TRAIN** SoftCtx MegaSuperExtra; R749+R753 N80 LIVE; B300×8=0; bal **~$87088.37** |
| 2026-08-18T05:41:12Z | p3807: **R751 REFUTE**→**R760 TRAIN** 6,7 pid**348814** + wait→merge+n80; R759 kept; B300×8=0; bal **~$87128.67** |
| 2026-08-18T05:36:22Z | p3806: **R749+R752+R753 wait→n80 ARMED**; R751~59/80; B300×8=0; bal **~$87170.07** |
