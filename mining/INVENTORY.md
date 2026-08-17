# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-17T19:41Z** | **R665** 4,5 · **R666** 6,7 · **R664** 2,3 · **R663** 0,1 · R654–R658 MERGE · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-17T19:41Z** | TK · R537 :8002 · **R651 SCP** 4,5 · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-17T19:41Z** | TK · R637 :8004 · **4,5 FREE** (R647 reaped) · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · **R662** 4,5 · **R668** 6,7 · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-17T19:41Z** | **R669** 0,1 · **R659/R660/R661** 2–7 · R651 uplink · R652/R653 MERGE · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · R596 :8002 · **R667 TRAIN** 6,7 · R655 MERGE · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R337**; B300×8 **empty**; burn **~$331.45/h** · **wvk=7**
**p3685:** R669 TRAIN brave 0,1 pid**209011**; R651~52G/12sh; bal **~$89459**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T15:41:51Z | p3685: **R669 TRAIN** brave 0,1 Long HiRank HiBeta ep3×LoLR pid**209011**; R651 SCP~11sh; B300 empty; bal **~$89459** |
| 2026-08-17T15:39:37Z | p3684: **R647 REFUTE** ~0.09× + reap golden 4,5; **R668 TRAIN** zesty 6,7 Long MidRank HiBeta ep3×LoLR pid**782729**; R651~38G/9sh; B300 empty; bal **~$89499** |
| 2026-08-17T15:34:06Z | p3683: **R667 TRAIN** R252 6,7 Soft HiRank HiBeta SoftCtx ep3×LoLR pid**252289**; R647 n80~31/80; R651~24G; B300 empty; bal **~$89539** |
