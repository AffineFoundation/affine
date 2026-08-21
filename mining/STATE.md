# STATE — mining run snapshot
Rewritten every pass. Do not append.

## Stage
**Stage 5 · wvk=7 Reason v4 · KING reign36 · R1064 LOST chal-00974**
King=**reign36** vera6 · burn floor **≥$833/h**.

## Live facts
| item | value |
|---|---|
| contract | wvk=**7** · k=**3** · τ=**0.03** · n=**1300** · δ=0.002 · thought≥80 · B γ=0.30 · k_σ=2.0 |
| king | `vera6/affine-5g4yy75zuz-t6`@`8e3f1695…` **reign36** |
| burn | **~$392.19/h** · gap **−$440.81/h** · B300×8=0 · **B200×8 stock=0** |
| Lium | **~$74667** · free τ**~1247** · stake **r252 ≈τ1.69** (<τ5 paygo) |
| fleet | **9** mine-* · TTL fleet **→2026-08-21T13:26Z** · r340 **→21:24Z** · r339 soft **14:56Z** |
| **p4276** | r926 **R1130** util0.85 KV−2.71→**TP1 util0.93 GPU7** chall+**n80** pid**168327**; r924 **R1132 REFUTE** ~−1.23× → **R1154** UltraLoLR TRAIN pid**159723** |
| eval | phase **duel** · **R1064 LOST** m=−0.000659 n=1286 |

## Running
| name | huid | $/h | role |
|---|---|---|---|
| mine-crown-1 | brave-comet-f4 | $64.00 | TK · **R1146+R1147+R1148 TRAIN** · SSH `95.133.252.28:40298` |
| mine-r252-vera-t4-nonking-grpo-1 | brave-wolf-f6 | $64.00 | TK · **R1138+R1153 TRAIN** · SSH `38.127.229.127:40299` |
| mine-r337-marsplan-online-dpo-hilr-1 | noble-hawk-1f | $46.80 | TK · **R1149+R1150 TRAIN** · SSH `150.136.46.118:20300` |
| mine-r338-marsplan-online-dpo-bigg-hilr-1 | calm-fox-6a | $52.25 | TK · **R1151+R1152 TRAIN** · SSH `95.133.253.90:40099` |
| mine-r339-marsplan-online-dpo-hirank-1 | noble-raven-a7 | $64.00 | TK · **R1143+R1145 TRAIN** · SSH `23.153.44.20:40299` |
| mine-r340-marsplan-online-dpo-hirank-bigg-1 | gentle-orbit-4a | $37.60 | TK · **R1142+R1141+R1144 TRAIN** · SSH `18.118.83.97:40127` |
| mine-r924-vera-midctx-hibeta-1 | cosmic-orbit-55 | $33.81 | TK · **R1131 n80 ~78/80** + **R1154 TRAIN** + R1139 TRAIN · SSH `31.22.104.113:40300` |
| mine-r926-cryptodev-softctx-midlobeta-1 | brave-raven-49 | $13.76 | TK · **R1130 n80 TP1 util0.93 GPU7** · SSH `93.120.231.186:32301` |
| mine-r938-vera-softctx-hibeta-1 | noble-wolf-22 | $15.96 | TK · **R1140 TRAIN** · SSH `38.255.28.21:20100` |

## Blocked
No rentable B300×8 / B200×8. Waiters armed (`wait_fleet_b300`). `lium fund` broken → `btcli wallet transfer` to Lium coldkey.

## Next action
1. Rent B300×8/B200×8 if stock. 2. Poll r926 **R1130** n80 (util0.93). 3. Harvest r924 **R1131** n80 (~78/80) → REFUTE→UltraLoLR. 4. Keep R1146–R1154 trains → MERGE→n80.
