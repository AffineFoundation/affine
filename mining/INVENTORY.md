# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK **tammy** · **R715 RELAY** 4,5 · **R737 TRAIN** 6,7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R731 TRAIN** 4,5 · **R728 N80** 6,7/:8003 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R729+R730 TRAIN** 4–7 |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R732+R733 TRAIN** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **idle** NCCL-blacklist; holds **R716** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R734+R735 TRAIN** 4–7 |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: **R728 N80 lunar** + **R715 relay** + R737/R729–R735 TRAIN; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3777:** R728 MERGE→CHALL→N80 LIVE lunar 6,7; R736 wait reaped; bal **~$87686**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T02:26:50Z | p3777: **R728 N80 LIVE** lunar 6,7/:8003 + R736 wait reaped; B300×8=0; bal **~$87686** |
| 2026-08-18T02:20:06Z | p3776: **brave NCCL abort** + **R715 relay→crown** (freed R736); B300×8=0; bal **~$87769** |
| 2026-08-18T02:09:00Z | p3775: **R727 REFUTE** + **R737 TRAIN** crown 6,7; R736 TRAIN 4,5; B300×8=0; bal **~$87850** |
