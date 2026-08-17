# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-17T19:41Z** | **R676–R678** · **R682 TRAIN** 6,7 · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-17T19:41Z** | TK · R537 :8002 · **R675 SCP** ~23G/6sh · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-17T19:41Z** | TK · R637 :8004 · **R681 SCP** ~53G/14sh · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · **R680** 6,7 · **R686 TRAIN** 4,5 · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-17T19:41Z** | **R685/R683/R687/R688 TRAIN** · **R679/R681** · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · R596 :8002 · **R684 TRAIN** 6,7 · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R337**; B300×8 **empty**; burn **~$331.45/h** · **wvk=7**
**p3713:** **R687+R688 TRAIN** brave 2–5; R681~14sh; R675~6sh; bal **~$87943**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T18:51:17Z | p3713: **R687+R688 TRAIN** brave 2–5 SoftHiBeta SoftCtx + ShortMidBeta ShortCtx UltraExtra; B300 empty; bal **~$87943** |
| 2026-08-17T18:47:28Z | p3712: **R686 TRAIN** zesty 4,5 Short HiRank LoBeta UltraExtra; freed r662+r674 merges; B300 empty; bal **~$87984** |
| 2026-08-17T18:42:15Z | p3711: **R663 REFUTE** −0.85× + **R674 REFUTE** −0.63×; **R681 SCP** brave→golden; B300 empty; bal **~$88024** |
