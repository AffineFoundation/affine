# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4/5 · wvk=7 Reason v4 · KING reign35 · R767 ARMED(p3836) · R783 TRAIN · R761 RELAY · R762 ARMED**.
King=**reign35** tammy · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `tammyfritz/Affine-5hmwhnfbix-tammy2`@`7e5fd5f8…` **reign35** |
| burn | **~$331.45/h** · gap **−$501.55/h** · B300×8 stock **0** |
| Lium | **~$86720** · free τ**1260.38** · stake **29.52α≈τ1.68** (below ~τ5 sweep) |
| **p3836** | **R767 MERGE idle→ARMED** resume-after-R762 pid**3158038** · **R783 TRAIN** brave 4,5 pid**293353** |
| **R761** | **RELAY** brave→R252 (~5/16 shards · ~21G/66G) · waiter GPUs **4,5** |
| **R762** | **WAIT_RELAY** after R761 → n80 4,5 after R761 decision |
| **R767** | **WAIT_RELAY** after R762 → n80 4,5 after R762 decision |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | gentle-orbit-bd | $52.25 | TK · **R775+R776 TRAIN** 4–7 |
| mine-r165-awesome-hialpha-1 | lunar-wolf-be | $44.00 | TK · **R779+R778 TRAIN** |
| mine-r262-kevin-v5-nonking-grpo-1 | golden-comet-78 | $60.00 | TK · **R774+R773 TRAIN** |
| mine-r260-elonmasky-ckp777-nonking-grpo-1 | zesty-comet-da | $64.00 | TK · **R772+R777 TRAIN** |
| mine-r226-marsplan-fullft-1 | brave-raven-a9 | $47.20 | **R783+R780+R781+R768 TRAIN**; R761/R762/R767→relay |
| mine-r252-vera-t4-nonking-grpo-1 | gentle-wolf-8c | $64.00 | TK · **R782 TRAIN** 6,7 · **R761 RELAY→N80** 4,5 |

## Blocked
No rentable 8×B300/B200 (bl ghost fbb1135f). R252 :40299 flaky — use `lium exec` / hardened host-relay. Never `pkill -f`. Hard-pin BASE after `mine.env`.

## Next action
1. R761 SCP_READY → chall+n80 R252 4,5. 2. R762 then R767 auto-relay → n80 after prior decision. 3. Watch R783/R781/R782→merge. 4. Rent when non-blacklisted B300/B200×8 appears. 5. Sweep α when ≥~τ5.
