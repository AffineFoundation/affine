# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R758+R753 TRAIN** · R753 wait→n80✓ |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R752+R749 TRAIN** · wait→n80✓ |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R755+R756 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R757+R754 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **idle** NCCL-blacklist |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R759 TRAIN** 4,5 · **R751 N80** 6,7/:8002 |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `lium exec gentle-wolf-8c` (direct :40299 flaky)
Host fleet: **R751 N80 ~59/80**; **R749/R752/R753 wait→n80 ARMED p3806**; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3806:** armed missing wait→n80 R749/R752/R753; bal **~$87170.07**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T05:36:22Z | p3806: **R749+R752+R753 wait→n80 ARMED**; R751~59/80; B300×8=0; bal **~$87170.07** |
| 2026-08-18T05:31:52Z | p3805: **R750 REFUTE**→**R759 TRAIN**; **R751 MERGE idle→n80 LIVE** chall**345330**; B300×8=0; bal **~$87209.37** |
| 2026-08-18T05:23:55Z | p3804: **R748+R716 REFUTE** → **R757+R758 TRAIN**; R750~75/80; bal **~$87291.25** |
