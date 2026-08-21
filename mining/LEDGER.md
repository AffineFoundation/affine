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
| $UTC | 75899.32915522243 | p4258 |
| Lium balance | **$75899.32915522243** | p4258 |
| cumulative mining spend | ~$148,338 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$65899** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.19/h** · vs floor $833/h · gap **−$440.81/h** · B300×8=0 · B200×8 stock=0 | p4258 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T04:32:48Z | 75899.32915522243 | p4258 **R1115 REFUTE→R1130 TRAIN** r926 GPUs3,4 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T04:23:46Z | 75977.46937066205 | p4257 **R1101 REFUTE→R1129 TRAIN** crown GPUs6,7 (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T04:11:26Z | 76053.07660059455 | p4256 **R1101 n80 re-arm** crown GPUs6,7 (no rent; B300/B200×8=0); R1115 N80 load; burn **~$392.19/h** |
| 2026-08-21T04:05:53Z | 76089.88955349677 | p4255 **R1128 TRAIN** r340 GPUs6,7 (no rent; B300/B200×8=0); R1115 MERGE; burn **~$392.19/h** |
| 2026-08-21T04:00:22Z | 76166.6817233292 | p4254 **R1108+R1109 REFUTE→R1126+R1127 TRAIN** r339 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:54:41Z | 76203.18614603757 | p4253 **R1106 REFUTE→R1125 TRAIN** r337 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:49:23Z | 76241.15718086746 | p4252 **R1105+R1102 REFUTE→R1123+R1124 TRAIN** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:43:20Z | 76280.53834000067 | p4251 **R1096+97+R1111 REFUTE→R1120+21+R1122 TRAIN** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:27:46Z | 76394.86134379286 | p4250 **R1107 REFUTE→R1119 TRAIN** r337 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:22:26Z | 76431.02864894563 | p4249 r340 **teacher65536 + R1096/R1097 n80 re-arm** (no rent; B300/B200×8=0); burn **~$392.18/h** |
