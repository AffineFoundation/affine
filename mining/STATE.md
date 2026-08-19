# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R861 LOST · R891/R874 REFUTE · R892+R893 MERGE→n80**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$405.70/h** · gap **−$427.30/h** · B300×8 stock=0 · 8×B200 stock=0 |
| Lium | **~$82358** · free τ**1257.6** · stake **~59α / τ3.25** (under τ5 sweep) |
| **R891** | **REFUTE v4** m=−0.010162~**−0.93×** thought✓(234) B✓(0.541) k=3 |
| **R874** | **REFUTE v4** m=−0.000952~**−0.22×** thought✓(201) B✓(0.425) k=3 · near-miss |
| **p3999** | R252 **R892+R893 TRAIN_DONE→MERGE LIVE** + wait→n80; lunar R896+R897 TRAIN |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R885+R886 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R896+R897 TRAIN** 4–7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R877+R878 TRAIN** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R889+R890+R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R892+R893 MERGE** 4–7 → n80 |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | TK **vera** · **R895 TRAIN** 4,5 · **R894 TRAIN** 6,7 |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · **R882–R884 TRAIN** |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R888 GRPO** ~step280 · GPU1/5/6 free |

## Blocked
No rentable B300/8×B200 (8×H200 only — skip). Marsplan HF gated — host-relay. Brave TP≥2 NCCL — TP=1. Never `pkill -f`. R888 shows **7**/8 B200s. Skip smoke urllib on Triton-risk challs. Do **not** re-arm R252 if `/root/logs/p3999_r892_r893_armed.done` exists. Do **not** re-arm lunar if `/root/logs/p3998_r896_r897_armed.done` exists.

## Next action
1. Poll R892/R893 MERGE→n80 verdicts; R896/R897 TRAIN→merge; R894/R895; R888 GRPO→merge. 2. Fill R888 idle GPUs 1/5/6 with a distinct axis. 3. Rent if B300 appears (skip fbb1135f).
