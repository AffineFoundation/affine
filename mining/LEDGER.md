# LEDGER — money in / money out

**Cap: 40 lines.** Totals + last 10 movements. Older → archive/.
Floor ≥ **$10,000**. Mining B300 burn floor **>$20,000/day = $833/h**.
Live burn = Σ $/h over `mine-*` pods every pass.

**Funding (operator 2026-08-16):** free to convert **SN120 Alpha → TAO → Lium**
(`lium fund -w miner`) to scale mining. Keep ≥ τ50 free coldkey for regs.
**SN120 α cannot pay Lium directly** — CLI only takes TAO or **SN51** α
(`lium fund --alpha …`). Optional SN51 probe OK; log every Alpha/TAO/Lium move.

## Totals

| item | value | as of |
|---|---|---|
| $UTC | 74263.05497162674 | p4281 |
| Lium balance | **$74263.05497162674** | p4281 |
| cumulative mining spend | ~$149,977 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$64263** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.19/h** · vs floor $833/h · gap **−$440.81/h** · B300×8=0 · B200×8 stock=0 | p4281 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T08:10:30Z | 74263.05497162674 | p4281 tore **1/8 GPU** r1158×2 (fbb1135f) + armed node-id waiter; burn **~$392.19/h** |
| 2026-08-21T08:00:48Z | 74332.63946079933 | p4280 **R1139 REFUTE**→**R1159** UltraLoLR + **r1158** bootstrap (no rent; B300/B200×8=0); burn **~$397.79/h** |
| 2026-08-21T07:47:51Z | 74445.4779214407 | p4279 **R1130 REFUTE**→**R1157** + **R1139** n80 + **rent** `mine-r1158` 8×B200 $5.60/h (brave-matrix-2a); burn **~$397.79/h** |
| 2026-08-21T07:32:55Z | 74594.06693887706 | p4278 r340 **R1141 REFUTE**→**R1156** Midβ UltraLoLR TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:26:51Z | 74594.06693887706 | p4277 r924 **R1131 REFUTE**→**R1155** UltraLoLR TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:21:28Z | 74666.51591855791 | p4276 r926 **R1130** util0.93 n80 + r924 **R1132 REFUTE**→**R1154** TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:09:18Z | 74741.02021943031 | p4275 r926 **R1130** TP2 OOM→**TP1 util0.85** n80 re-arm (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:05:50Z | 74741.02021943031 | p4274 r252 **R1123 REFUTE**→**R1153** UltraLoLR TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:00:17Z | 74813.88144756506 | p4273 r337 **R1125+37 REFUTE**→**R1149+50** + r338 **R1135+36 REFUTE**→**R1151+52** TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T06:52:18Z | 74850.81148337232 | p4272 crown **R1129+33+34 REFUTE**→**R1146+47+48** TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
