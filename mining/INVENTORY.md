# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R758+R763 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R752 MERGE→n80** + **R764 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R755+R756 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R757+R754 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R761+R762 TRAIN** SoftCtx Mega |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R759+R760 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | golden-lion-72 | 8×B200 | $45.60 | **~24h from 05:53Z** | **R337 BOOT** Online-DPO HiLR |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `lium exec gentle-wolf-8c` (direct :40299 flaky)
SSH R337: `ssh -p 20296 root@192.9.163.79` / `lium exec golden-lion-72`
Host fleet: **7 mine-*** · R337 claimed · B300×8 **0** · burn **~$377.05/h** · **wvk=7**
**p3810:** waiter rented **mine-r337** `golden-lion-72` 8×B200 $45.60; HF marsplan gated on pod IP → zesty→gold cache copy + wait→boot; bal **~$86916.46**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T06:06:22Z | p3810: **R337 claim** golden-lion-72 + marsplan cache copy/wait→boot; B300×8=0; bal **~$86916.46** |
| 2026-08-18T05:55:23Z | p3809: **R753+R749 REFUTE**→**R763+R764 TRAIN** wait→merge+n80; B300×8=0; bal **~$87005.70** |
| 2026-08-18T05:47:40Z | p3808: **brave idle→R761+R762 TRAIN** SoftCtx MegaSuperExtra; R749+R753 N80 LIVE; B300×8=0; bal **~$87088.37** |
