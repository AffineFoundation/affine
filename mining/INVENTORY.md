# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R956+R958 n80** · R957 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | **R960+R951 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | **R963+R954 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | **R955 n80** · R959 TRAIN · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · **R961 TRAIN** · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | R953+R952 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | **cold-TK FAIL** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-20T21:23Z** | T+K · **R962 TRAIN** · SSH `38.255.28.21:20100` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$329.78/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T03:02:40Z | p4080: R956+R958+R955 MERGE idle→**v4 n80 LIVE**; R938 ~313/1300; B300×8=0 BL `fbb1135f` |
| 2026-08-20T02:54:40Z | p4079: R949 **REFUTE**→reap→**R963 TRAIN** R337 4,5; R938 chal-00949 scoring; B300×8=0 |
| 2026-08-20T02:47:03Z | p4078: R949 MERGE idle→**v4 n80 LIVE** R337 :8002; R926 teacher Triton miss→FAIL; B300×8=0 BL `fbb1135f` |
