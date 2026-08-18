# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R717/R718 TRAIN** 4–7 · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R712 TRAIN** 4,5 · **R721 TRAIN** 6,7 · keep `/tmp/r675_merged` · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R719/R720 TRAIN** 4–7 · keep r683/r637 · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R722 TRAIN** 4,5 · **R723 TRAIN** 6,7 · keep r708/r709 · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R713–R716 TRAIN** 0–7 · kept r696/699/700/704 · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R710/R711 TRAIN** 4–7 · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R722/R723 train→merge**; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3758:** R708/R709 REFUTE → R722+R723 TRAIN zesty 4–7; bal **~$85379**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T00:06:23Z | p3758: **R708+R709 REFUTE**; reap zesty chall; **R722+R723 TRAIN** MidCtx Mid/HiBeta SuperExtra; B300×8=0; bal **~$85379** |
| 2026-08-17T23:55:44Z | p3757: reap lunar idle **r537** chall → **R721 TRAIN**; B300×8=0; bal **~$85461** |
| 2026-08-17T23:48:24Z | p3756: **R709+R708 N80 LIVE** zesty 4–7; B300×8=0; bal **~$85542** |
