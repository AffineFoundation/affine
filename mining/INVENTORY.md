# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R811+R812 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R807+R808 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R804 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R796+R797** (SSH dead) |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R809+R810 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R805+R806 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | (check) | **R337** form wait sim |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: **:20299 refused** — retry later / console
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: `lium exec gentle-wolf-8c` / `ssh -p 40299 root@95.133.252.28`
SSH R337: `lium exec gentle-shark-35`
Host fleet: **7 mine-*** · B300×8 rentable **0** · burn **~$378.50/h** · **wvk=7**
**p3885:** R801 REFUTE→R811+R812 TRAIN; α→TAO→Lium; bal **~$87868**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T14:54:30Z | p3885: **R801 REFUTE** ~0.072× → free :8003 → **R811+R812 TRAIN**; 160α→τ8.9→Lium; B300×8=0; bal **~$87868** |
| 2026-08-18T14:44:06Z | p3884: **R801** SIZE_OK→lean **:8003** LOAD; B300×8=0; bal **~$86196** |
| 2026-08-18T14:34:39Z | p3883: **R800 REFUTE** ~−0.46× → free 4,5 → **R809+R810 TRAIN** brave; R801 ACCEL; B300×8=0; bal **~$86275** |
