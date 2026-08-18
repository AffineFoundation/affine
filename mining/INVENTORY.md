# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK **tammy** · **R715 N80** 4,5/:8002 · **R744 TRAIN** 6,7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R741 TRAIN** 4,5 · **R738 TRAIN** 6,7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R745+R746 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R740 TRAIN** 4,5 · **R739 N80** 6,7/:8003 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **idle** NCCL-blacklist; holds **R716** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R743+R742 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: **R715+R739 N80 LIVE** + 8× TRAIN; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3786:** R739 MERGE→**N80 LIVE** zesty 6,7/:8003; bal **~$87117**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T03:40:08Z | p3786: **R739 MERGE_DONE→N80 LIVE** zesty 6,7/:8003; R715~30/80; B300×8=0; bal **~$87117** |
| 2026-08-18T03:36:13Z | p3785: **R715 SCP_READY**→**N80 LIVE** crown 4,5/:8002; B300×8=0; bal **~$87117** |
| 2026-08-18T03:17:30Z | p3784: **5× REFUTE** → **R742–R746 TRAIN**; R715~48G; B300×8=0; bal **~$87280** |
