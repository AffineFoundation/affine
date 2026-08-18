# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING FLIP reign35 · R722–R725 N80 LIVE · R726–R731 TRAIN · R719–R721 REFUTE · R713–R716 MERGE idle brave**.
King=**reign35** tammy · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `tammyfritz/Affine-5hmwhnfbix-tammy2`@`7e5fd5f8…` **reign35** |
| burn | **~$331.45/h** · gap **−$501.55/h** · B300×8 stock **0** |
| Lium | **~$88133** · free τ**1260.38** · r252 stake **0** |
| **R720** | **REFUTE v4** m=−0.001125~**−0.33×** (thought✓145 B✓0.403) Short MidRank MidBeta SuperExtra · chall reaped · keep `/tmp/r720_merged` |
| **R721** | **REFUTE v4** m=−0.000446~**−0.05×** (thought✓206 B✓0.439) marsplan MidCtx LoBeta SuperExtra · chall reaped · keep `/tmp/r721_merged` |
| **R722** | **N80 LIVE** zesty 4,5/:8002 ~80/80 scoring `*_reign35_wvk7` |
| **R723** | **N80 LIVE** zesty 6,7/:8003 ~80/80 scoring `*_reign35_wvk7` |
| **R724** | **N80 LIVE** R252 4,5/:8002 ~80/80 scoring `*_reign35_wvk7` |
| **R725** | **N80 LIVE** R252 6,7/:8003 ~80/80 scoring `*_reign35_wvk7` |
| **R729** | **TRAIN** MidCtx MidRank HiBeta SuperExtra golden 4,5 |
| **R730** | **TRAIN** Short HiRank MidBeta SuperExtra golden 6,7 pid**544629** + wait→merge |
| **R726/R727** | **TRAIN** crown 4–7 |
| **R728** | **TRAIN** lunar 6,7 |
| **R731** | **TRAIN** MidCtx LoBeta HyperExtra lunar 4,5 pid**693370** + wait→merge |
| **R713–R716** | **MERGE_DONE** brave (no TK) — next SCP/TK or train fill |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK tammy · **R726+R727 TRAIN** |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK tammy · **R731+R728 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK · **R729+R730 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da | $64.00 | TK · **R722+R723 N80** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | R713–R716 MERGE idle · no TK |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK · **R724+R725 N80** |

## Blocked
No 8×B300 (waiters armed). Never `pkill -f`. Next n80s **must** use reign35. Hard-pin `GPUS`/`CHALL_PORT` after `mine.env`.

## Next action
1. Harvest **R722–R725** (`*_reign35_wvk7.json`) → WIN→Stage5 / REFUTE→reap+next train. 2. Watch **R726–R731** train→merge. 3. Fill brave (TK bring or SCP R713–R716). 4. Rent B300 when stock.
