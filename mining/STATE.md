# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R861 LOST · R890+R887+R889 REFUTE · R905+R906+R907 TRAIN · R903+R904 TRAIN · R901+R902 TRAIN · R899+R900+R898 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$405.70/h** · gap **−$427.30/h** · B300×8 stock=0 · 8×B200 stock=0 (1×B200 only) |
| Lium | **~$81852** · free τ**1257.6** · stake **~59α / τ3.25** (under τ5 sweep) |
| **R890** | **REFUTE v4** m=−0.00290 SE=0.00197 bar=0.00395 (~**−0.73×**) thought✓170 B✓0.419 |
| **R887** | **REFUTE v4** m=−0.00193 SE=0.00308 bar=0.00615 (~**−0.31×**) thought✓182 B✓0.418 |
| **R889** | **REFUTE v4** m=−0.00638 SE=0.00303 bar=0.00606 (~**−1.05×**) thought✓172 B✓0.455 |
| **p4006** | triple REFUTE→**R905+R906+R907** TRAIN brave · `p4006_r905_armed.done` + `p4006_r906_r907_armed.done` |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R901+R902 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R896+R897 TRAIN** 4–7 |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R903+R904 TRAIN** 4–7 |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** TP1 · **R905+R906+R907 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK **vera** · **R899+R900 TRAIN** 4–7 |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | TK **vera** · **R895+R894 TRAIN** 4–7 |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | teacher · **R882–R884 MERGE** idle (host-relay) |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king · **R898 TRAIN** 5,6 |

## Blocked
No rentable B300/8×B200 (1×B200 only — skip). Marsplan HF gated — host-relay. Brave TP≥2 NCCL — TP=1. Never `pkill -f`. Do **not** re-arm R905 if `/root/logs/p4006_r905_armed.done` exists. Do **not** re-arm R906/R907 if `/root/logs/p4006_r906_r907_armed.done` exists. Do **not** re-arm R903/R904 if `/root/logs/p4005_r903_r904_armed.done` exists. Do **not** re-arm R901/R902 if `/root/logs/p4004_r901_armed.done` / `p4004_r902_armed.done` exists.

## Next action
1. Poll R905/R906/R907 TRAIN→merge→n80. 2. Poll R903/R904/R901/R902/R899/R900/R898 TRAIN. 3. R338 MERGE idle (R882–R884) → host-relay or local n80 if slots free. 4. Rent if B300×8 appears (skip fbb1135f).
