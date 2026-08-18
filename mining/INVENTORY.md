# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · R717/R718 · **tammy 16/16 swap** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · R712+R721 · prefetch **6/16** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · R719/R720 · prefetch **3/16** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | R722+R723 · **tammy 16/16 swap** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | R713–R716 TRAIN · no TK |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | R724+R725 TRAIN · **tammy 16/16 swap** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `ssh -p 40299 root@95.133.252.28` / `lium exec gentle-wolf-8c`
Host fleet: king→**reign35 tammy** retarget LIVE; B300×8 **0**; burn **~$331.45/h** · **wvk=7**
**p3763:** α→TAO→Lium τ16.73 · bal **~$88582**; retarget crown/R252/zesty 16/16 swap

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T00:37:29Z | p3763: **α→TAO→Lium** τ16.73; retarget 16/16 swap on 3 TK; lunar/golden still DL; bal **~$88582** |
| 2026-08-18T00:32:21Z | p3762: **reign35** tammy; retarget :8001 on 5 TK; HF tok refresh; B300×8=0; bal **~$85217** |
| 2026-08-18T00:26:04Z | p3761: **R711 REFUTE**; **R725 TRAIN**; B300×8=0; bal **~$85257** |
