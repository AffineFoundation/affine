# INVENTORY — live mine-* pods

**Cap: 40 lines.** Live table + last 3 reconciles. Older → `archive/`.

## Live

| name | huid | config | $/h | TTL | role |
|---|---|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | 8×B300 | $64.00 | **2026-08-20T18:12Z** | TK · R929 TRAIN · R928 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | 8×B200 | $64.00 | **2026-08-20T18:14Z** | R3 GRPO MERGE→n80 · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | 8×B200 | $46.80 | **2026-08-20T18:14Z** | vera online-DPO · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | 8×B200 | $52.25 | **2026-08-20T18:14Z** | **R338 n80 + R935 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | 8×B200† | $39.20 | **2026-08-20T03:34Z** | T+king · **R932 TRAIN** · SSH `192.9.163.79:20500` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | 8×H200 | $33.81 | **2026-08-20T18:41Z** | R924+R925+R930+R931 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | 8×H100 | $13.76 | **2026-08-20T19:18Z** | R926+R927+R933+R934 TRAIN · SSH `93.120.231.186:32301` |

†nvidia-smi shows **7** GPUs. Host fleet: **7 mine-*** · burn **~$313.82/h** · **wvk=7**

Non-mine (do not touch): `affine-bench`, `affine-datagen`, `affine-teacher`, `affine-eval`, `affine-teacher2`.

## Last reconciles

| when | action |
|---|---|
| 2026-08-19T20:49:30Z | p4040: R338 KING_READY+chall; fill idle GPUs **6,7** → **R935 TRAIN** HiRank Loβ MidCtx; stock BL-only `8f34559f`; burn ~$313.82/h |
| 2026-08-19T20:43:28Z | p4039: R338 TRAIN_DONE merge abort (marsplan BASE) → **vera merge→king→n80** armed; stock BL-only; burn ~$313.82/h |
| 2026-08-19T20:34:56Z | p4038: stock BL-only `8f34559f`; fill R926 idle GPUs **4–7** → **R933+R934 TRAIN**; burn ~$313.82/h |
