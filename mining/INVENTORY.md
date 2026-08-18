# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R800 RELAY** 4,5 · **R801 ARMED** 6,7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R807+R808 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R804+R803 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R796+R797** (SSH dead) |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R801 MERGE** src · R800 src |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R805+R806 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: **:20299 refused** — retry later / console
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: `lium exec gentle-wolf-8c` / `ssh -p 40299 root@95.133.252.28`
Host fleet: **6 mine-*** · B300×8 rentable **0** · burn **~$331.45/h** · **wvk=7**
**p3880:** R783 REFUTE→R801 armed; R800 mid-pipe (~3/16+); bal **~$86509**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T14:04:46Z | p3880: **R783 REFUTE** ~0.12× → free 6,7 → **R801** wait R800.done; B300×8=0; bal **~$86509** |
| 2026-08-18T13:59:16Z | p3879: **R783 SIZE_OK→n80** + **R800 fast×4**; B300×8=0; bal **~$86546** |
| 2026-08-18T13:50:00Z | p3878: **R800** MERGE_DONE→defer relay after R783; R783 ~12/16; B300×8=0; bal **~$86616** |
