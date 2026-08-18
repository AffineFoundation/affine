# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R811+R812 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R817+R808 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R813+R814 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R809+R810 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R815+R816 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T14:05Z** | **p3893** fill + **p3894** meta✓ |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T15:01Z** | **p3893** fill + **p3894** meta✓ |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: `ssh -p 40299 root@95.133.252.28`
SSH R337: `ssh root@86.38.182.67 -p 20295`
SSH R338: `ssh root@86.38.182.55 -p 20299`
Host fleet: **7 mine-*** · B300×8 rentable **0** · lone 8×B200 **bl** · burn **~$366.49/h** · **wvk=7**
**p3894:** meta prestage R337/R338 (cfg✓ 16 shards); blobs still piping; bal **~$87385**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T15:51:37Z | p3894: **meta prestage** R337/R338 while p3893 pipes; mine=7; burn **~$366.49/h** |
| 2026-08-18T15:46:07Z | p3893: **continuous×8 fill** R337/R338 (kill p3892 parents; keep pipes); mine=7; burn **~$366.49/h** |
| 2026-08-18T15:40:18Z | p3892: **dual accel** R337/R338 (kill p3891 parent only; keep pipes); mine=7; burn **~$366.49/h** |
