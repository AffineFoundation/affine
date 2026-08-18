# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK **tammy** · **R753 TRAIN** 6,7 · **R716 RELAY** 4,5 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R752+R749 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R745+R746 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R748 TRAIN** 4,5 · **R747 N80** 6,7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **idle** NCCL-blacklist; R716 source |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R750+R751 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `lium exec gentle-wolf-8c` (direct :40299 flaky)
Host fleet: **R747 N80** + TRAIN cascade + R716 RELAY; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3797:** **R747 MERGE→N80 LIVE** zesty 6,7 :8003 chall**892881**; R716 ~11/16; bal **~$87617**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T04:42:11Z | p3797: **R747 MERGE→N80 LIVE** zesty 6,7; B300×8=0; bal **~$87617** |
| 2026-08-18T04:36:31Z | p3796: **R744 REFUTE** m=−0.001936~−0.43× → **R753 TRAIN** crown 6,7; B300×8=0; bal **~$87698** |
| 2026-08-18T04:32:00Z | p3795: **α→TAO→Lium** r252 **89.96α→τ5.11**; bal **~$86668→$87698**; B300×8=0; R744 ~60/80 |
