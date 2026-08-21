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
| $UTC | 75370.30197069787 | p4265 |
| Lium balance | **$75370.30197069787** | p4265 |
| cumulative mining spend | ~$148,867 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$65370** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.19/h** · vs floor $833/h · gap **−$440.81/h** · B300×8=0 · B200×8 stock=0 | p4265 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T05:44:49Z | 75370.30197069787 | p4265 **R1121 REFUTE**→**R1141** UltraLoLR TRAIN r340 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T05:38:25Z | 75408.89117020546 | p4264 **R1121** TP1 n80 re-arm; **R1114→R1139** + **R1118→R1140** TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T05:26:39Z | 75483.71254345735 | p4263 **R1121** stuck chall→n80 re-arm r340; purge r926+r252 merges (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T05:18:49Z | 75560.11677698362 | p4262 **R1110→R1138** UltraLoLR TRAIN r252 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T05:13:30Z | 75597.57419847416 | p4261 **R1122+24→R1135+36** + **R1119→R1137** (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T05:01:37Z | 75712.66405303222 | p4260 **R1112+13→R1131+32** + **R1116+17→R1133+34** (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T04:41:54Z | 75825.70068784968 | p4259 crown **/tmp ENOSPC** cleanup + **R1116+R1117 n80 re-arm** (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T04:32:48Z | 75899.32915522243 | p4258 **R1115 REFUTE→R1130 TRAIN** r926 GPUs3,4 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T04:23:46Z | 75977.46937066205 | p4257 **R1101 REFUTE→R1129 TRAIN** crown GPUs6,7 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T04:11:26Z | 76053.07660059455 | p4256 **R1101 n80 re-arm** crown GPUs6,7 (no rent; B300/B200×8=0); R1115 N80 load; burn **~$392.19/h** |
