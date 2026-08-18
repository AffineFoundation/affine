# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK · **vera DIRECT×4 ~62%** · R829 n80 LIVE · R827 wait |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK LOAD · **R837 TRAIN** · **R836 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK · **4,5 FREE** · **R828 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | **R821–R823** + relay |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | **R834+R833 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | **R835 TRAIN** + **R796+R830** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R338 merge-only** + **R831+R832** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: `ssh -p 40299 root@95.133.252.28`
SSH R337: `ssh root@86.38.182.67 -p 20295`
SSH R338: `ssh root@86.38.182.55 -p 20299`
Host fleet: **7 mine-*** · B300×8 rentable **0** · lone 8×B200 **bl** · burn **~$366.49/h** · **wvk=7**
**p3925:** R824 REFUTE→R829 n80 LIVE; vera DIRECT×4 ~62%; golden 4,5 FREE; B300×8=0; mine=7; bal **~$85437**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T19:34:06Z | p3925: **R824 REFUTE→R829 n80 LIVE** crown :8002; vera ~62%; golden 4,5 FREE; B300×8=0 |
| 2026-08-18T19:27:18Z | p3924: **R826 REFUTE→R837** + vera DIRECT×4 + lunar TK relaunch; R825 REFUTE reap; B300×8=0 |
| 2026-08-18T19:16:56Z | p3923: **crown vera** kill HF + lunar→crown shard1 pipe; R826/R825 n80 LIVE; B300×8=0 |
