# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-17T19:41Z** | **R676–R678** · **R682 TRAIN** 6,7 · **R663 uplink** · **v4 pkg** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-17T19:41Z** | TK · R537 :8002 · **R675 SCP** 4,5/:8003 (~2G/1sh) · **v4 pkg** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-17T19:41Z** | TK · R637 :8004 · **R663 SCP** 4,5 (~49G/13sh) · **v4 pkg** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · **R680** 6,7 · **R674** 4,5 · R668 keep · **v4 pkg** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-17T19:41Z** | **R669 MERGE** · **R679/R681** · **R683 TRAIN** 6,7 · R670–R672 keep · **v4 pkg** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · R596 :8002 · **R675 uplink** · **R684 TRAIN** 6,7 · **v4 pkg** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R337**; B300×8 **empty**; burn **~$331.45/h** · **wvk=7**
**p3707:** **R684 TRAIN** R252 6,7 pid**262017**; R675~2G/1sh; R663~49G/13sh; bal **~$88310**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T18:07:06Z | p3707: **R684 TRAIN** R252 6,7 MidCtx MidRank HiBeta UltraExtra (β=0.3 r=32 @8192 steps=7200) pid**262017** + wait→merge; B300 empty; bal **~$88310** |
| 2026-08-17T18:03:36Z | p3706: **R655 REFUTE** m=−0.00390~−0.92× (k=3); **R675 SCP** R252→lunar (+repair); reaped lunar 4,5; B300 empty; bal **~$88348** |
| 2026-08-17T17:57:30Z | p3705: **R655 N80_LIVE** ~55/80; **R682+R683 TRAIN** crown/brave 6,7 UltraExtra; freed crown REFUTE r641+r643; B300 empty; bal **~$88391** |
