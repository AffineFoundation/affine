# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · **R972+R971+R970 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | TKC · **R960+R951 v4 n80 LIVE** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | TK · **R954+R963 MERGE idle** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | TK · **R959+R964 MERGE idle** · SSH `95.133.253.90:40099` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | **R969+R968 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | TKC · **R944 v4 n80 ~50/80** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | 4×H200 | $15.96 | **2026-08-20T21:23Z** | T+K · **R962 MERGE idle** · SSH `38.255.28.21:20100` |

Host fleet: **7 mine-*** (R888 gone) · burn **~$290.58/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-20T06:02:18Z | p4097: R252 R960+R951 MERGE idle→**dual v4 n80 LIVE** :8002/:8003; B300×8=0 |
| 2026-08-20T05:52:32Z | p4096: R944 teacher OOM@TP2/0.88→**TP4@0.85**→**v4 n80 LIVE** pid58462; B300×8=0 |
| 2026-08-20T05:41:09Z | p4095: R944 teacher DOWN→TP2@0.88→**v4 n80 LIVE** pid55267; R888 absent; B300×8=0 |
