# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R861 LOST · R866 n80 LIVE · R863–R865 MERGE · R874/R875/R876 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$366.50/h** · gap **−$466.50/h** · B300×8=0 · B200×8=[] |
| Lium | **~$84714.56** · free τ**1257.6** · stake **0** |
| **R861** | **LOST** chal-00934 m=+0.001182 SE=0.000646 z=1.83 n=1288 bar=δ**0.002** (~**0.59×**) thought✓169 B✓0.416 k=3 |
| **p3972** | R338 p3954 merge outer failed (`gpus_p3954` missing digits) → **R863/R864/R865 MERGE LIVE** (pids 62086/62088/62092); R866 n80 ~74/80 |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · R861 LOST · R862 idle |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R866 n80** ~74/80 · **R874 TRAIN** 6,7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · R858/R859 idle REFUTE challs |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R875+R876 TRAIN** 4–7 |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R872/R873 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | teacher · R869–R871 MERGE_DONE · GPUs 2–7 idle |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · **R863/R864/R865 MERGE LIVE** 2–7 |

## Blocked
No rentable B300/B200×8 (JSON=[]). Marsplan HF gated — host-relay only. Host→R252:40299 timeout — `lium exec`/`lium scp`. Brave TP≥2 NCCL-spins — TP=1. Never `pkill -f`. HF public storage tight.

## Next action
1. Poll R866 n80 DONE→CLEAR?→Stage5 (fail-closed k=3). 2. Wait R863–R865 MERGE_DONE→host-relay→n80. 3. Arm wait→merge R874/R875/R876 (+R872/R873). 4. Reap golden idle challs / fill R337. 5. Rent if B300/B200×8 appears.
