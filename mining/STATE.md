# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R861 LOST · R863 REFUTE · R864 n80 · R888 teacher reload**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$405.70/h** · gap **−$427.30/h** · B300×8 stock=0 · 8×B200 stock=0 |
| Lium | **~$84081.87** · free τ**1257.6** · stake **0** |
| **R861** | **LOST** chal-00934 m=+0.001182~**0.59×**δ thought✓ B✓ k=3 |
| **R863** | **REFUTE v4** m=−0.00584~**−0.63×** thought✓215 B✓0.338 k=3 |
| **p3981** | R864 **CHALL_READY+n80 LIVE** ~31/80; R888 teacher DeepGEMM crash → patch `VLLM_USE_DEEP_GEMM=0` + relaunch pid**6086** LOAD → wait→GRPO |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R885+R886 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R864 n80 LIVE** · **R874 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R877+R878 TRAIN** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R875+R876+R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R872/R873 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | teacher · **R879–R881 TRAIN** 2–7 |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · **R882–R884 TRAIN** |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | **R888** teacher LOAD (deep_gemm=0) → TP1 READY → GRPO |

## Blocked
No rentable B300/8×B200. Marsplan HF gated — host-relay. Brave TP≥2 NCCL — TP=1. Never `pkill -f`. HF public storage tight. R888 shows **7**/8 B200s. R888 teacher needs **DeepGEMM off** on this image.

## Next action
1. Poll R864 n80 DONE→CLEAR?→Stage5 (fail-closed k=3) else reap→R865 relay. 2. Poll R888 TEACHER_READY→`start_r888` GRPO TRAIN (host poll `.ralph/p3981_poll_r888_teacher*`). 3. Poll R887/R885/R886/R879–R884 TRAIN→merge→n80. 4. Rent if B300 appears (skip fbb1135f).
