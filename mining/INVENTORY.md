# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R798+R799 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R790+R789 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R792:4,5 + R791:6,7 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R796+R797 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R794+R795 TRAIN**; R783/R784 MERGE_DONE; R780/R781 WAIT_RELAY |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R768 PARALLEL SCP+meta** · **R793 TRAIN** · R767 idle |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` (lium exec flaky) / `lium exec gentle-wolf-8c`
Host fleet: **6 mine-*** · B300×8 rentable **0** · burn **~$331.45/h** · **wvk=7**
**p3856:** R768 **meta accel** cfg✓ (pid**3521920**); weight parallel×4 KEEP; bal **~$85499**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T11:10:15Z | p3856: **R768 meta accel** cfg✓ while weight×4; B300×8=0; bal **~$85499** |
| 2026-08-18T11:06:31Z | p3855: **R786/R788 REFUTE→R798+R799 TRAIN** crown; B300×8=0; bal **~$85499** |
| 2026-08-18T10:57:04Z | p3854: **R768 SCP stall→parallel×4** KEEP 5+8; B300×8=0; bal **~$85581** |
