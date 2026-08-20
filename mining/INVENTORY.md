# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R965+R966+R967 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | **R960+R951 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | **R963+R954 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | **R959+R964 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · **R961 TRAIN** · SSH `192.9.163.79:20500` **refused** |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R953+R952 v4 n80 ~33/29**/80 · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | TKC · **R944 v4 n80 LIVE** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-20T21:23Z** | T+K · **R962 TRAIN** · SSH `38.255.28.21:20100` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$329.78/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T04:59:26Z | p4090: R926 teacher **TP=2@0.88** READY→probe→**R944 v4 n80 LIVE** pid48964; R924 dual n80 ~33/29; B300×8=0 |
| 2026-08-20T04:49:16Z | p4089: R924 R952+R953 MERGE idle→**dual v4 n80 LIVE** (pids 38112/37705); R926 T DOWN (p4088c FATAL); B300×8=0 |
| 2026-08-20T04:13:05Z | p4087: R944 Triton hang→FORCE king seed→probe n_so=22→**v4 n80 LIVE** pid38459; R924 R952/R953 MERGE idle; R888 :20500 refused; B300×8=0 |
