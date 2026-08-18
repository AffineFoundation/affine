# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R839+R840 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R837 TRAIN** · **R842 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK tammy→**vera RELAY** · **R838+R841 TRAIN** |
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
**p3928:** R836 REFUTE→R842 TRAIN; golden vera DIRECT LIVE; B300×8=0; mine=7; bal **~$85038**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T20:17:40Z | p3928: **R836 REFUTE** reap→**R842 TRAIN**; lunar→golden vera DIRECT×6 LIVE; B300×8=0 |
| 2026-08-18T20:06:15Z | p3927: **R828 REFUTE** reap→**R841 TRAIN**; R836 n80 LIVE vs vera; B300×8=0 |
| 2026-08-18T19:56:07Z | p3926: **R829+R827 REFUTE** reap→**R838** golden + **R839/R840** crown; vera SIZE_OK+SWAP; B300×8=0 |
