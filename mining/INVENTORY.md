# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd (`3d07e519-…`) | 8×B200 | $52.25 | **2026-08-18T19:04Z** | TK · **R775+R776 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be (`0f283151-…`) | 8×B200 | $44.00 | **2026-08-18T19:04Z** | TK · **R779+R778 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 (`522c84fd-…`) | 8×B200 | $60.00 | **2026-08-18T19:04Z** | TK · **R774+R773 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da (`5e54186c-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | TK · **R772+R777 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 (`5c019a27-…`) | 8×B200 | $47.20 | **2026-08-18T19:04Z** | **R781+R780+R767+R768 TRAIN**; R762→relay |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c (`61d17753-…`) | 8×B300 | $64.00 | **2026-08-18T19:04Z** | **R782 TRAIN** + **R761 RELAY** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `lium exec lunar-wolf-be` / `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH R260: `lium exec zesty-comet-da` / `ssh root@86.38.182.95 -p 20299`
SSH brave: `ssh -p 40127 root@18.118.83.97` / `lium exec brave-raven-a9`
SSH R252: `lium exec gentle-wolf-8c` (direct :40299 flaky) / `ssh -p 40299 root@95.133.252.28`
Host fleet: **6 mine-*** · B300×8 rentable **0** · burn **~$331.45/h** · **wvk=7**
**p3834:** R762 resume-after-R761 armed pid**3129501**; R761 ~13G; bal **~$86841**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval` (eager-fox-27), `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-18T08:24:51Z | p3834: **R762 ARMED** resume-after-R761 host-relay pid**3129501**; R761 ~13G/3shards; B300×8=0; bal **~$86841** |
| 2026-08-18T08:20:32Z | p3833: **R770 REFUTE→R782 TRAIN** R252 6,7; **R781 TRAIN** brave 2,3; R761 RELAY ~9.3G; B300×8=0; bal **~$86841** |
| 2026-08-18T08:10:37Z | p3832: **R769 REFUTE→R780 TRAIN** brave 0,1; R761 RELAY ~6G; B300×8=0; bal **~$86922** |
