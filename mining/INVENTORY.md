# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-17T19:41Z** | cold TK · R643/R641 tar src |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-17T19:41Z** | TK · **R643 SCP** →4,5/:8003 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-17T19:41Z** | TK · **R648 SCP**~58G/13sh →4,5/:8003 |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · **R641 SCP** →4,5 · R634→6,7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-17T19:41Z** | **R651** 0,1 + **R652** 2,3 · R648 uplink |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-17T19:41Z** | TK · R596 :8002 · **R631 DEFER** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: next=**R337**; B300×8 **empty**; burn **~$331.45/h**
**p3656:** R652 TRAIN brave 2,3; bal **~$90753**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-17T13:02:09Z | p3656: **R652 TRAIN** Soft HiRank LoBeta ep3×LoLR brave **2,3** pid**187664**; R651 alive; R648~58G/13sh; B300 empty; bal **~$90753** |
| 2026-08-17T12:55:53Z | p3655: **R631 STALL** @12G flat → kill+**DEFER** (wait R648; host pid**428475**); R648~26G R641~48G R643~23G; R651~step200; B300 empty; bal **~$90796** |
| 2026-08-17T12:51:23Z | p3654: **R645 REFUTE** m=−0.001633~−0.29×; **R648 ARMED** golden (reap R645; brave→golden relay); R651~step130; B300 empty; bal **~$84361** |
