# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-17T19:41Z** | **R676 TRAIN** 0,1 · **R665** 4,5 · **R673** 6,7 · **R664** 2,3 · R663 MERGE · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-17T19:41Z** | TK · R537 :8002 · **R655 SCP+REPAIR→chall** 4,5/:8003 · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-17T19:41Z** | TK · R637 :8004 · **R653 n80 LIVE** 4,5/:8003 · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · R662 MERGE · **R668** 6,7 · **R674 TRAIN** 4,5 · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-17T19:41Z** | **R669–R672 TRAIN** · R653 uplink done · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · R596 :8002 · R667 MERGE · **R675 TRAIN** 6,7 · **R655 uplink** · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R337**; B300×8 **empty**; burn **~$331.45/h** · **wvk=7**
**p3698:** R676 TRAIN crown pid**25063**; R653 n80 ~41/80; R655 SCP+repair; API waiter **945012**; bal **~$88798**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T17:05:37Z | p3698: **R676 TRAIN** crown 0,1 after R663 MERGE; R653 n80~41/80; B300 empty; bal **~$88798** |
| 2026-08-17T17:00:34Z | p3697: **R653 SCP_READY→CHALL** (16sh/67G exact); R655~10sh/40G; B300 empty; bal **~$88880** |
| 2026-08-17T16:54:51Z | p3696: **R653+R655 REPAIR** armed (full size-check post-tar; R653 missing shard4); B300 empty; bal **~$88892** |
