# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R861 LOST · R852/R853 REFUTE · R835+R796 n80**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$405.70/h** · gap **−$427.30/h** · B300×8 stock=0 · 8×B200 stock=0 |
| Lium | **~$82697** · free τ**1257.6** · stake **~59α / τ3.25** (under τ5 sweep) |
| **R852** | **REFUTE v4** m=−0.006589~**−0.81×** thought✓(250) B✓(0.464) k=3 |
| **R853** | **REFUTE v4** m=−0.007160~**−0.63×** thought✓(203) B✓(0.429) k=3 |
| **p3996** | R852+R853 REFUTE→reap→**R835 n80 LIVE** :8003; **R796 n80 LIVE** :8002 (repair after double-arm) |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R885+R886 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R891 MERGE_DONE** idle 4,5 · **R874** EngineDead 6,7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R877+R878 TRAIN** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R889+R890+R887 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R892+R893 TRAIN** 4–7 |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | TK **vera** · **R835 n80** :8003 · **R796 n80** :8002 |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · **R882–R884 TRAIN** |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R888 GRPO** · GPU5 free |

## Blocked
No rentable B300/8×B200 (8×H200 only — skip). Marsplan HF gated — host-relay. Brave TP≥2 NCCL — TP=1. Never `pkill -f`. R888 shows **7**/8 B200s. Skip smoke urllib on Triton-risk challs. Do **not** re-run dual-arm if `/root/logs/p3996b_r835_r796_armed.done` exists.

## Next action
1. Poll R835 + R796 n80 CLEAR/REFUTE. 2. On REFUTE: reap exact PID → next axis (not SoftCtx HiRank rematch). 3. Lunar: **R891** MERGE_DONE → chall+n80 GPUs4,5; Triton seed+RELOAD R874 GPUs6,7. 4. Poll R892/R893 TRAIN→merge→n80; R888 GRPO→merge; fill GPU5. 5. Rent if B300 appears (skip fbb1135f).
