# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | 8×B200 | $52.25 | **2026-08-19T17:23Z** | TK **vera** · **R861+R862 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | 8×B200 | $44.00 | **2026-08-19T17:23Z** | TK **vera** · **R866+R860 TRAIN** · R854 slot-wait |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | 8×B200 | $60.00 | **2026-08-19T17:23Z** | TK **vera** · **R859+R858 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | 8×B200 | $47.20 | **2026-08-19T17:23Z** | **TK vera TP1 · R848+R849 n80 LIVE** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | 8×B300 | $64.00 | **2026-08-19T17:23Z** | TK **vera LIVE** · **R867+R868 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | 8×B300 | $47.04 | **2026-08-19T17:35Z** | teacher · **R869–R871 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | 8×B300 | $52.00 | **2026-08-19T17:35Z** | **R863–R865 TRAIN** |

SSH crown: `ssh root@95.133.253.90 -p 40099`
SSH R165: `ssh -p 20299 root@150.136.46.118`
SSH R262: `ssh root@38.127.229.127 -p 40299`
SSH brave: `ssh -p 40127 root@18.118.83.97` (host-key churn after reboot)
SSH R252: **host:40299 TIMEOUT** — use `lium exec gentle-wolf-8c` / `lium scp -d`
SSH R337: `ssh root@86.38.182.67 -p 20295`
SSH R338: `ssh root@86.38.182.55 -p 20299`
Host fleet: **7 mine-*** · B300×8=0 · B200×8=`fbb1135f` bl · burn **~$366.49/h** · **wvk=7**
**p3965:** R849 n80 relaunch pid**21643** (corpus race); R848 pid**20779**; mine=7

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T01:55:00Z | p3965: R849 corpus FATAL→n80 relaunch :8003; R848 LIVE; stock bl-only |
| 2026-08-19T01:51:00Z | p3964: brave R848/R849 TP1 READY + n80 LIVE; stock 0 |
| 2026-08-19T01:45:17Z | p3963: brave TP2 king NCCL→TP1 READY; R848/R849 n80 armed; stock 0 |
