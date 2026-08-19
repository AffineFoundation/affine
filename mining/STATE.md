# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R861 LOST · R863 REFUTE · R864 relay LIVE · R887 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$366.50/h** · gap **−$466.50/h** · B300×8=0 · B200×8 listed (skip bl) |
| Lium | **~$84361.58** · free τ**1257.6** · stake **0** |
| **R861** | **LOST** chal-00934 m=+0.001182~**0.59×**δ thought✓ B✓ k=3 |
| **R863** | **REFUTE v4** m=−0.00584~**−0.63×** thought✓215 B✓0.338 k=3 |
| **p3977** | R863 REFUTE→reap 4,5 → **R864 host-relay LIVE**; brave idle→**R887 MidLoβ TRAIN** |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R885+R886 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R864 relay→n80** · **R874 TRAIN** 6,7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R877+R878 TRAIN** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R875+R876+R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R872/R873 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | teacher · **R879–R881 TRAIN** 2–7 · R869–R871 MERGE parked |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · **R882–R884 TRAIN** 2–7 · R864/R865 MERGE (R864 relay) |

## Blocked
No rentable B300; lone 8×B200=`fbb1135f` bl. Marsplan HF gated — host-relay only. Host→R252:40299 timeout — `lium exec`/`lium scp`. Brave TP≥2 NCCL-spins — TP=1. Never `pkill -f`. HF public storage tight.

## Next action
1. Poll R864 SIZE_OK→CHALL_READY→n80 DONE→CLEAR?→Stage5 (fail-closed k=3). 2. Host-relay R865 (+R869–R871) after R864 frees lunar 4,5. 3. Poll R887 TRAIN→merge→n80. 4. Poll R885/R886 TRAIN→merge→n80. 5. Poll R879–R884 TRAIN→merge. 6. Arm wait→merge R874/R875/R876 (+R872/R873). 7. Poll R877/R878 TRAIN→merge→n80. 8. Rent if B300 appears (skip fbb1135f).
