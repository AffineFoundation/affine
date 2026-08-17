# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R693 TRAIN** 4,5 · **R692 TRAIN** 6,7 · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R675 SCP** ~64G/16sh · repair armed · waiter→chall · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · R637 :8004 · **R680 SCP** ~30G/8sh · **R683 queued** · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R694 TRAIN** 4,5 · R691 TRAIN · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | R685/R687/R690 · **R696 TRAIN** 4,5 · **R688/R683 MERGE_DONE** · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R695 TRAIN** 4,5 · R689 TRAIN 6,7 · R675 src · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R338**; B300×8 **empty**; burn **~$331.45/h** · **wvk=7**
**p3725:** **R696 TRAIN** pid**241280** brave 4,5 (fill after R688 MERGE); R680~30G R675~64G; bal **~$87333**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T20:07:52Z | p3725: **R696 TRAIN** fill idle brave 4,5 after R688 MERGE; B300 empty; bal **~$87333** |
| 2026-08-17T20:02:15Z | p3724: **R695 TRAIN** reaped R596 chall→R252 4,5 Short HiRank MidBeta UltraExtra; B300 empty; bal **~$87371** |
| 2026-08-17T19:56:50Z | p3723: **R683 SCP_QUEUED** gate R680→golden; R680~21G R675~58G; B300 empty; bal **~$87414** |
