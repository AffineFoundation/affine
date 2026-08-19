# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 5 · wvk=7 Reason v4 · KING reign36 · R861 SUBMITTED · R850/R851 + R867/R868 n80 LIVE**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$366.49/h** · gap **−$466.51/h** · B300×8=0 · B200×8=0 rentable |
| Lium | **~$84936.87** · free τ**1257.6** · stake **0** |
| **R861** | **SUBMITTED** HF `@f2075358…` · reg **8875721-0018** · reveal **31434087** · block `0x8f9a17fc…` |
| **p3969** | brave R850/R851 chall failed flashinfer JIT → reload w/ `VLLM_USE_FLASHINFER_*=0` → **n80 LIVE** |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · R861 submitted · R862 idle |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R860/R866 TRAIN** · **R854** slot-wait |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · R858/R859 idle post-REFUTE |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R850:8002 + R851:8003 n80 LIVE** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R867/R868 n80** ~40/80 |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | teacher · R869–R871 merged · GPUs 2–7 idle |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · R863–R865 · GPUs 2–7 idle |

## Blocked
No rentable B300/B200×8. Marsplan HF gated — host-relay only. Host→R252:40299 timeout — `lium exec`/`lium scp`. Brave TP≥2 NCCL-spins — **all** TK+chall use TP=1. Brave chall must set `VLLM_USE_FLASHINFER_SAMPLER=0` (+ sibling flags) or flashinfer ninja dies on cu13. Never `pkill -f`. HF public storage tight — purge LOST merges before next push.

## Next action
1. Poll brave R850/R851 + R252 R867/R868 n80 results (fail-closed k=3). 2. Watch R861 duel (reveal **31434087**). 3. On REFUTE: free slot → next axis same pass. 4. Reap golden R858/R859 → next axes; fill R337/R338 idle. 5. Rent if non-bl B300/B200×8 appears.
