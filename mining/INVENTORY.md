# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R784+R783 SCP→n80** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R790+R789 REFUTE** idle |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R804+R803 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R796+R797 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R800+R801 TRAIN** · **R794/R795 rematch** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R805+R802 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `lium exec gentle-wolf-8c` (SSH :40299 flaky) / `ssh -p 40299 root@95.133.252.28`
Host fleet: **6 mine-*** · B300×8 rentable **0** · burn **~$331.45/h** · **wvk=7**
**p3870:** R798+R799 REFUTE→R784+R783 relay; brave ENOSPC free+R794/R795 rematch; bal **~$84603**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T12:55:32Z | p3870: **R798+R799 REFUTE** → **R784+R783 host-relay**; brave ENOSPC→**R794/R795 rematch**; B300×8=0; bal **~$84603** |
| 2026-08-18T12:47:25Z | p3869: **R792+R781 REFUTE** → **R804** golden 4,5 + **R805** R252 4,5 TRAIN; B300×8=0; bal **~$84724** |
| 2026-08-18T12:39:05Z | p3868: **R781** SIZE_OK→stamp + kill−9 STOP parent; chall loading :8002; B300×8=0; bal **~$84765** |
