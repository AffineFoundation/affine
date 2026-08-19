# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4 · wvk=7 Reason v4 · KING reign36 · R914 n80 LIVE + fleet bootstrap p4025**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$266.26/h** · gap **−$566.74/h** · B300×8 stock=0 · 8×B200 stock=0 (bl-only) |
| Lium | **~$79409** · free τ**1257.6** · stake **0** |
| fleet | 5 mine-* · stale `.bootstrapped` cleared · R252/R337/R338 upload OK |
| **R914** | MERGE OK · **n80 relaunched** outer pid**33797** chall pid**33895** :8002 GPUs5,6 |
| **crown** | empty · waiter stamped `needs_axis_uploader` (no mine-crown-1 case) |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | **EMPTY** · needs TK/axis uploader · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | bootstrap_r3 LIVE · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | bootstrap_h139 LIVE · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | bootstrap LIVE · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king warm · R888 GRPO · **R914 n80 LOADING** |

## Blocked
No fresh B300×8 / non-blacklisted 8×B200. Never rent `8f34559f…` (MoE hang). Never `pkill -f`.
`mine-crown-1` has **no** `wait_bootstrap` case → needs dedicated TK uploader.

## Next action
1. Poll R914 n80 → Stage-5 iff margin>max(2·SE,δ) ∧ thought≥80 ∧ B≥0.30. 2. Crown TK bootstrap (add uploader / serve_three). 3. After R252/R337/R338 boot: re-arm stranded Offline-DPO (R912–R922). 4. Rent if B300/B200 stock appears.
