# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R846+R847 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R843 TRAIN** · **R860 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R859+R858 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | **R848–R851 TRAIN** 0–7 |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | **R844+R845 TRAIN** · vera part0/1 GROW |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | **R852–R854 TRAIN** 2–7 |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R855–R857 TRAIN** 2–7 |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: **host:40299 TIMEOUT** — use `lium exec gentle-wolf-8c` / `lium scp`
SSH R337: `ssh root@86.38.182.67 -p 20295`
SSH R338: `ssh root@86.38.182.55 -p 20299`
Host fleet: **7 mine-*** · B300×8=0 · B200×8=**1 listed=bl** · burn **~$366.49/h** · **wvk=7**
**p3950:** R830 REFUTE→R860 TRAIN; vera part1 resume GROW; mine=7

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T23:23:30Z | p3950: **R830 REFUTE→R860 TRAIN** lunar 6,7; vera part1 resume; B200 bl-skip |
| 2026-08-18T23:12:10Z | p3949: **R830 EngineDead→RELOAD** lunar :8003; B200×8 listed (not rented); B300=0 |
| 2026-08-18T23:08:14Z | p3948: R252 vera **part1 resume** + kill stale dual-writers; lium-exec SIZE_OK concat; B300=0 B200=0 |
