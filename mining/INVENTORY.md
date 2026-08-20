# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · R943+R945+R946 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | **R942 TRAIN** GPUs4,5 · T+king · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | R939 + **R941 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | **R947 TRAIN** + **R935 n80:8003** · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · R932 TRAIN · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R925/R930 MERGE_DONE · R931 MERGE** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | **R944 TRAIN** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 8×H200 | $15.96 | **2026-08-20T21:23Z** | R938 TRAIN · SSH `38.255.28.21:20100` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$329.79/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T23:59:43Z | p4059: R937 **REFUTE** ~0.27× → reap :8002 → **R947 TRAIN** + **R935 n80:8003**; R924 **R925/R930 MERGE_DONE** R931 mid; B300=0 |
| 2026-08-19T23:50:13Z | p4058: R934 **REFUTE** ~−0.34× → reap :8002 → **R946 TRAIN** + **R937 n80** R338 MERGE_DONE; B300=0 |
| 2026-08-19T23:38:36Z | p4057: R926 **REFUTE** ~−0.80× → reap :8003 → **R944**+**R945 TRAIN**; R934 SIZE_OK→lean:8002 **n80 LIVE**; B300=0 |
