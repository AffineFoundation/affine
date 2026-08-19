# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4 · wvk=7 Reason v4 · KING reign36 · R912/R913 MERGE→n80 + crown TK DL + R926/R927 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$313.82/h** · gap **−$519.18/h** · B300×8=0 · non-bl B200×8=0 (only **BL** `fbb1135f`) |
| Lium | **~$78892** · free τ**1257.6** · stake **0** |
| fleet | 7 mine-* · crown R912+R913 MERGE + TK DL · R923 · R924+R925 · R926+R927 TRAIN · R337/R338 · R252 GRPO |
| **R912** | TRAIN DONE ~900 steps → **MERGE** GPUs 6,7 · n80 waiter armed · needs TK :8000/:8001 |
| **R913** | TRAIN DONE ~884 steps → **MERGE** GPUs 4,5 · n80 waiter armed · needs TK |
| **R926** | **TRAIN** SoftCtx MidLoβ GPUs 0,1 pid**3143** (cryptoDev DL done) |
| **R927** | **TRAIN** MidCtx MidLoβ GPUs 2,3 pid**3581** (hardened arm: 16 shards+evalsrv import) |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | R912+R913 MERGE · TK teacher DL · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | R3 GRPO TRAIN · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | vera online-DPO TRAIN · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | vera online-DPO BigG TRAIN · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R923 TRAIN** GPUs 5,6 |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | **R924+R925 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | **R926+R927 TRAIN** · SSH `93.120.231.186:32301` |

## Blocked
No rentable B300×8 / non-bl 8×B200 (only BL `fbb1135f`). Marsplan queen **gated**. Never `pkill -f`.

## Next action
1. Poll crown TK READY (:8000/:8001) then R912/R913 MERGE→n80. 2. Poll R926/R927/R924/R925/R923→merge→n80. 3. Rent if non-bl B300/B200 stock; replace H100/H200 when B300 appears.
