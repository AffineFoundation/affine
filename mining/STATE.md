# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign36 · R882 REFUTE · R899 CHALL_READY · R884 n80 arm · R883 Triton orphan cleanup · R908+R909 TRAIN**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$405.70/h** · gap **−$427.30/h** · B300×8 stock=0 |
| Lium | **~$81344** · free τ**1257.6** · stake under τ5 sweep |
| **R882** | **REFUTE v4** m=**−0.007812** SE=0.003789 z=−2.06 n=80 bar≈0.00758 (~**−1.03×**) thought✓(226) B✓(0.439) · slot→**R884** |
| **R899** | **n80 LIVE** p4012c R252 pid**610406** (outer died after CHALL_READY; sim relaunched) |
| **R884** | chall loading R338 :8002 after R882 reap · n80 waiter armed |
| **R883** | Triton fail → orphan workers; cleanup may have hit EngineCores — **verify T/K :8000/:8001** next (SSH flapped) |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK **vera** · **R901+R902 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK **vera** · **R896+R897 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK **vera** · **R903+R904 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | TK **vera** · **R905+R906+R907 TRAIN** |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | **R899 n80** + **R900 TRAIN** |
| mine-r337-marsplan-online-dpo-hilr-1 | gentle-shark-35 | $47.04 | **R908+R909 TRAIN** |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-lion-9f | $52.00 | **R884** + **R883** repair · SSH `86.38.182.55:20299` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | **R898 MERGE** |

## Blocked
No B300×8 stock. **R338 (`calm-lion-9f` / 86.38.182.55:20299) unreachable** — TCP timeout + `lium exec` timeout while status=RUNNING; sibling R337 SSH also timed out (same /24). Next: retry SSH/exec; if still dead, `lium rm` **only** that `mine-*` and re-rent axis (R883/R884 merges live on disk until then). Never `pkill -f`. Never broad-kill `VLLM::EngineCore`.

## Next action
1. Retry R338 SSH/`lium exec`; restore T/K if up. 2. Poll R899 n80 (CHALL_READY). 3. If R338 stays dead → tear+replace mine-r338 only. 4. Rent if B300×8 appears.
