# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK **tammy** · **R726+R727 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK **tammy** · **R731+R728 TRAIN** 4–7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK **tammy** · **R729+R730 TRAIN** 4–7 |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R722+R723 N80** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | R713–R716 MERGE idle · no TK |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R724+R725 N80** 4–7 |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: **R722–R725 N80** + **R726–R731 TRAIN**; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3770:** R720+R721 REFUTE → R730+R731 TRAIN; R722–R725 N80 still scoring 80/80

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T01:32:14Z | p3770: **R720+R721 REFUTE** + **R730+R731 TRAIN**; B300×8=0; bal **~$88133** |
| 2026-08-18T01:25:30Z | p3769: **R719 REFUTE** + **R722–R725 N80** + **R729 TRAIN**; B300×8=0; bal **~$88216** |
| 2026-08-18T01:15:45Z | p3768: **R712 REFUTE** + **R719/R720/R721 N80** + **R728 TRAIN**; B300×8=0; bal **~$88297** |
