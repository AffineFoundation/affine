# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK **tammy** · **R715 RELAY** 4,5 · **R737 N80** 6,7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R731 N80** 4,5 · **R738 TRAIN** 6,7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R729 N80** 4,5 · **R730 MERGE** 6,7 |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R740 TRAIN** 4,5 · **R739 TRAIN** 6,7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **idle** NCCL-blacklist; holds **R716** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R734+R735 N80 LIVE** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: **5× n80 arming/live** + R715 relay ~35G; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3782:** armed **R737+R729** lean n80 (MERGE_DONE idle slots); bal **~$87400**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T03:01:00Z | p3782: **R737+R729 N80 ARMING**; R734/5/731 LIVE; R715~35G; B300×8=0; bal **~$87400** |
| 2026-08-18T02:54:57Z | p3781: **R734+R735+R731 N80 LIVE**; R715 ~31G/66G; B300×8=0; bal **~$87484** |
| 2026-08-18T02:50:20Z | p3780: **R732 REFUTE** + **R740 TRAIN** zesty 4,5; R739 kept 6,7; B300×8=0; bal **~$87484** |
