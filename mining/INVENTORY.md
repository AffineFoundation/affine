# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R810 ACCEL** → n80 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R817+R818 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R813+R814 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R821–R824 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R820+R819 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T14:05Z** | **R337 TRAIN** online-DPO |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T15:01Z** | **R338 TRAIN** online-DPO BigG |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH brave: `ssh -p 40127 root@18.118.83.97`
SSH R252: `ssh -p 40299 root@95.133.252.28`
SSH R337: `ssh root@86.38.182.67 -p 20295`
SSH R338: `ssh root@86.38.182.55 -p 20299`
Host fleet: **7 mine-*** · B300×8 rentable **0** · lone 8×B200 **bl** · burn **~$366.49/h** · **wvk=7**
**p3904:** R809 REFUTE→free :8002; R810 tail×6 ACCEL; brave **R821–R824 TRAIN**; mine=7; burn **~$366.49/h**; bal **~$86853**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T16:53:58Z | p3904: **R809 REFUTE~−0.39×** free :8002; **R810** ×6 ACCEL; brave **R821–R824 TRAIN**; B300×8=0 |
| 2026-08-18T16:46:11Z | p3903: **R815 REFUTE~−0.30×** → **R820 TRAIN**; **R809 SIZE_OK→chall**; **R810 defer RELAY**; B300×8=0 |
| 2026-08-18T16:39:54Z | p3902: **R816 REFUTE~−0.63×** → free :8003 → **R819 TRAIN**; R815 n80 LIVE; R809 mid |
