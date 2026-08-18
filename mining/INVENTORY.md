# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R783 RELAY** · 4,5 free |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R795 N80** · **R807 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R804+R803 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R796+R797** (SSH dead) |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R800+R801 TRAIN** · src |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R805+R806 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: **:20299 refused** — retry later / console
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: `lium exec gentle-wolf-8c` / `ssh -p 40299 root@95.133.252.28`
Host fleet: **6 mine-*** · B300×8 rentable **0** · burn **~$331.45/h** · **wvk=7**
**p3876:** R802/R794/R784 REFUTE→R806+R807 TRAIN; R795 ~48/80; bal **~$86688**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T13:37:42Z | p3876: **R802/R794/R784 REFUTE** → R806+R807 TRAIN; R795 n80; B300×8=0; bal **~$86688** |
| 2026-08-18T13:28:57Z | p3875: R784 **SIZE_OK**→stamp→lean :8002; R783 relay; B300×8=0; bal **~$86758** |
| 2026-08-18T13:21:06Z | p3874: **SIGSTOP p3871** → R784 tail+meta; R783-only after; B300×8=0; bal **~$86793** |
