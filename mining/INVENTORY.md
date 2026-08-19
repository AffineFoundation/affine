# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · R929/R936 TRAIN · **R924 n80 arm** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | R3 GRPO MERGE→n80 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | **R337 king→n80** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | R935+R937 TRAIN · T+king · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · **R932 TRAIN** · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | MERGE_DONE · R925/R930/R931 · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | R926/R933/R934 TRAIN · R927 adapter · SSH `93.120.231.186:32301` |

†nvidia-smi shows **7** GPUs. Host fleet: **7 mine-*** · burn **~$313.82/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T21:16:47Z | p4044: R924 MERGE_DONE→**host-relay→crown** chall:8002 GPUs6,7 n80 arm; stock B300=0 BL B200 only; H200×8 `$31.92` free |
| 2026-08-19T21:10:19Z | p4043: R337 TRAIN_DONE→marsplan merge abort → **vera MERGE→n80** pid**15785**; R924 wait FATAL→**merge** GPUs0,1; stock BL-only |
| 2026-08-19T21:03:07Z | p4042: R338 **REFUTE** ~−0.35× → reap chall:8002 → **R937** SoftCtx HiRank Midβ; stock BL-only `8f34559f` |
