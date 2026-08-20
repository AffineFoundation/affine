# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · R943+R945+R946 n80 · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | R942+R951 TRAIN · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | R949 + **R954 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | **R947 n80** + **R955 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · R950 TRAIN · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | R953+R952 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | R944 TRAIN · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-20T21:23Z** | post-submit · SSH `38.255.28.21:20100` |

†nvidia-smi shows **7** GPUs. Host fleet: **8 mine-*** · burn **~$329.79/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T01:40:03Z | p4071: R938 HF@`8ef1b06a` **SUBMITTED** reveal 31462190; R941/R948 REFUTE→reap→**R954+R955 TRAIN** + **R947 n80**; B300=0 |
| 2026-08-20T01:30:43Z | p4070: R938 **CROWN_OK** ~1.20× → hotkey+reg+HF push; R941/R948 MERGE→chall+n80; B300=0 |
| 2026-08-20T01:20:14Z | p4069: crown R943/R945/R946 MERGE idle → triple chall+n80; R938 n80 LIVE; B300=0 |
