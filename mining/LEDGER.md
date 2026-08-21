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
| $UTC | 74741.02021943031 | p4275 |
| Lium balance | **$74741.02021943031** | p4275 |
| cumulative mining spend | ~$149,497 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$64741** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.19/h** · vs floor $833/h · gap **−$440.81/h** · B300×8=0 · B200×8 stock=0 | p4275 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T07:09:18Z | 74741.02021943031 | p4275 r926 **R1130** TP2 OOM→**TP1 util0.85** n80 re-arm (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:05:50Z | 74741.02021943031 | p4274 r252 **R1123 REFUTE**→**R1153** UltraLoLR TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:00:17Z | 74813.88144756506 | p4273 r337 **R1125+37 REFUTE**→**R1149+50** + r338 **R1135+36 REFUTE**→**R1151+52** TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T06:52:18Z | 74850.81148337232 | p4272 crown **R1129+33+34 REFUTE**→**R1146+47+48** TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T06:43:47Z | 74923.61928045264 | p4271 **R1127 REFUTE**→**R1145** TRAIN r339 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T06:34:49Z | 74995.48447529739 | p4270 **R1126+R1128 REFUTE**→**R1143+R1144** TRAIN + **R1127** n80 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T06:21:39Z | 75070.99839400969 | p4269 **R1128** OOM@0.90→**TP1 util0.85** n80 r340 GPU6 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T06:14:45Z | 75144.96330853707 | p4268 **R1128** TP2 stall→**TP1** n80 r340 GPU6 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T06:04:00Z | 75214.7841515009 | p4267 **R1120 REFUTE**→**R1142** UltraLoLR TRAIN r340 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T05:54:14Z | 75295.22889675791 | p4266 **R1120** TP2 stall→**TP1** n80 r340 (no rent; B300/B200×8=0); burn **~$392.19/h** |
