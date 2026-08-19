# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 4 · wvk=7 Reason v4 · KING reign36 · FLEET REBUILD p4024**.
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$266.26/h** · gap **−$566.74/h** · B300×8 stock=0 · 8×B200 stock=0 (bl-only) |
| Lium | **~$79442** · free τ**1257.6** · stake **0** |
| fleet | TTL-collapse ~17:23Z killed 5 mine-*; only R888 survived → **rented 4** |
| **R914** | MERGE SIZE_OK on R888 · n80 **FATAL** chall ConnectError@11:08Z → **relaunch n80** |
| **R918–R921 / R912–R917 / R911→R922** | hosts **DEAD** (TTL) · adapters may be lost · re-arm after bootstrap |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | **NEW 8×B300** · bootstrap pending · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | **NEW 8×B200** · bootstrap pending · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | **NEW 8×B200** · Online-DPO HiLR · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | **NEW 8×B200** · Online-DPO BigG×HiLR · SSH `95.133.253.90:40099` |
| mine-r888-grpo-reason-1 | gentle-orbit-0d | $39.20 | T+king warm · R888 GRPO · **R914 merge ready / n80 re-arm** |

## Blocked
No fresh B300×8 / non-blacklisted 8×B200. Never rent `8f34559f…` (MoE hang). Never `pkill -f`.

## Next action
1. Bootstrap crown/R252/R337/R338 (wait_bootstrap armed). 2. Relaunch R914 lean_chall n80 on R888 GPUs5,6. 3. Rent if B300/B200 stock appears. 4. Re-arm stranded Offline-DPO axes after TK warm.
