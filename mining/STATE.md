# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4 · wvk=7 Reason v4 · KING reign36 · R923 n80 RUNNING · R929/R928 TRAIN · R926/R927/R924/R925 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$313.82/h** · gap **−$519.18/h** · B300×8=0 · only BL 8×B200 `fbb1135f` |
| Lium | **~$81194** · free τ**1258.1** · stake **0** (r252 swept p4036) |
| fleet | 7 mine-* · crown TK + R929/R928 TRAIN · R923 n80 · R924+R925 · R926+R927 · R337/R338 · R252 GRPO |
| **R923** | **n80 RUNNING** R888 :8002 READY · sim pid**42129** · out `r923_sim_result_reign36_wvk7.json` |
| **R929** | **TRAIN** HiRank MidLoβ ShortCtx r=64 β=0.05 GPUs 4,5 pid**31471** |
| **R928** | **TRAIN** HiRank Midβ MidCtx GPUs 6,7 pid**30385** |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK READY · R929 TRAIN · R928 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | R3 GRPO TRAIN · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | vera online-DPO TRAIN · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | vera online-DPO BigG TRAIN · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R923 n80 RUNNING** :8002 · GRPO 2,3 |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | **R924+R925 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | **R926+R927 TRAIN** · SSH `93.120.231.186:32301` |

## Blocked
No rentable B300×8 / non-bl 8×B200 (`fbb1135f` BL). Marsplan queen **gated**. Never `pkill -f`. `lium fund` broken (`Subtensor.transfer` missing) → use `btcli wallet transfer` to Lium coldkey.

## Next action
1. Poll R923 n80 → submit if clears; else REFUTE→free→next TRAIN. 2. Poll R929/R928 TRAIN→merge→n80. 3. Poll R926/R927/R924/R925. 4. Rent if non-bl B300/B200 stock.
