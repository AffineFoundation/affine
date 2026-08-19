# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4 · wvk=7 Reason v4 · KING reign36 · R927 ARMED + R926 DL + R925/R924/R912/R913/R923 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$313.82/h** · gap **−$519.18/h** · B300×8=0 · non-bl B200×8=0 (only **BL** `fbb1135f`) |
| Lium | **~$78965** · free τ**1257.6** · stake **0** |
| fleet | 7 mine-* · crown R912+R913 · R923 · R924+R925 · R926 SoftCtx + **R927 MidCtx** · R337/R338 · R252 GRPO |
| **R927** | **ARMED** brave-raven-49 GPUs **2,3** wait `cryptodev_dl.done`+weights →TRAIN MidCtx MidLoβ (affine_pkg fixed) |
| **R926** | **DL** cryptoDev after pip OK · then TRAIN SoftCtx MidLoβ GPUs **0,1** |
| **R925** | TRAIN cosmic-orbit-55 GPUs **2,3** HiRank MidCtx MidLoβ |
| **R924** | TRAIN GPUs **0,1** MidCtx Hiβ · wait→merge armed |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | R912+R913 TRAIN · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | R3 GRPO TRAIN · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | vera online-DPO TRAIN · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | vera online-DPO BigG TRAIN · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R923 TRAIN** GPUs 5,6 |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | **R924+R925 TRAIN** · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | **R926 DL+TRAIN + R927 ARMED** · SSH `93.120.231.186:32301` |

## Blocked
No rentable B300×8 / non-bl 8×B200 (only BL `fbb1135f` cosmic-raven-04 $44.8). Marsplan queen **gated**. Never `pkill -f`.

## Next action
1. Poll R927 arm→TRAIN after cryptoDev DL; R926 SoftCtx TRAIN. 2. Poll R925/R924/R912/R913/R923→merge→n80. 3. Rent if non-bl B300/B200 stock; replace H100/H200 when B300 appears. 4. Crown TK on 0–3 after teacher DL for n80.
