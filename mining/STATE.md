# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 5 · wvk=7 Reason v4 · KING reign36 · R861 CLEAR→HF push LIVE · R848/R849 n80 LIVE · R860/R866 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$366.49/h** · gap **−$466.51/h** · B300×8=0 · B200×8=`fbb1135f` **bl** only |
| Lium | **~$85111.62** · free τ**1260+** · stake **0** (p3966 swept) |
| **R861** | n80 **CLEAR** m=**+0.003665** ~**1.088×** bar (SE=0.001684 thought✓141.5 B✓0.5375 k=3) → HF push LIVE · hotkey `r861`=`5FBwq…69aA` |
| **p3966** | Stage5 armed; α284→τ15.63→Lium; R858/R859/R862 REFUTE noted |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R861 HF push** · R862 chall idle post-REFUTE |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R860/R866 TRAIN** · **R854** slot-wait |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · R858/R859 chall idle post-REFUTE |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera TP1** · **R848+R849 n80 LIVE** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · R867/R868 merged · GPUs 4–7 idle |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | teacher · R869–R871 merged · GPUs 2–7 idle |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · R863–R865 train/merge · GPUs 2–7 idle |

## Blocked
No rentable B300; lone 8×B200=`fbb1135f` **bl**. Marsplan HF gated — host-relay only. Host→R252:40299 timeout — `lium exec`/`lium scp -d`. Brave TP≥2 NCCL-spins — **all** TK+chall use TP=1. Never `pkill -f`.

## Next action
1. Poll crown `r861_hf_pushed.done` → register `r861` → `submit.py --check` → submit. 2. Poll brave R848/R849. 3. Reap R858/R859/R862 → next axes; arm R867/R868 n80 on R252. 4. Rent if non-bl B300/B200×8 appears.
