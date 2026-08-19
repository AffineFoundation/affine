# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R861 LOST · R894+R895 REFUTE · R908+R909 TRAIN · R905+R906+R907 TRAIN · R903+R904 TRAIN · R901+R902 TRAIN · R899+R900+R898 TRAIN · R882+R883 n80 ARMING**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$405.70/h** · gap **−$427.30/h** · B300×8 stock=0 · 8×B200 stock=0 |
| Lium | **~$81545** · free τ**1257.6** · stake **~59α / τ3.25** (under τ5 sweep) |
| **R894** | **REFUTE** m=**−0.006518** ~−0.78× thought✓200 B✓0.573 → slot **R908** |
| **R895** | **REFUTE** m=**−0.008260** ~−0.88× thought✓197 B✓0.468 → slot **R909** |
| **R908+R909** | **TRAIN** R337 GPUs 6,7 / 4,5 pids **126336** / **126371** · wait→merge |
| **R882+R883** | R338 local king DL ~91% (28G incomplete) → dual n80; **R884** slot-waiter |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R901+R902 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R896+R897 TRAIN** 4–7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R903+R904 TRAIN** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R905+R906+R907 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R899+R900 TRAIN** 4–7 |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | TK **vera** · **R908+R909 TRAIN** 4–7 |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | T · king DL · **R882+R883 n80 arm** + **R884 wait** |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R898 TRAIN** 5,6 |

## Blocked
No rentable B300/8×B200. Marsplan HF gated — local n80 on R338. Never `pkill -f`. Do **not** re-arm R908/R909 if `/root/logs/p4009_r908_r909_armed.done` exists. Do **not** re-arm R882/R883 if `/root/logs/p4008_r882_r883_armed.done` exists. Do **not** re-arm R884 if `/root/logs/p4008_r884_armed.done` exists.

## Next action
1. Poll R338 king DL→R882/R883 n80 + R884. 2. Poll R908+R909 TRAIN→merge. 3. Poll TRAIN fleet. 4. Rent if B300×8 appears.
