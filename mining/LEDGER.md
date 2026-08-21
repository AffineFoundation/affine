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
| $UTC | 73379.25539080291 | p4295 |
| Lium balance | **$73379.25539080291** | p4295 |
| cumulative mining spend | ~$150,860 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$63379** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.19/h** · vs floor $833/h · gap **−$440.81/h** · B300×8=0 · B200×8=BL-only | p4295 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T10:14:48Z | 73379.25539080291 | p4295 **r340 R1142+R1144+R1156 n80 RELAUNCH** TP1/0.85 (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T10:04:54Z | 73449.71999398018 | p4294 **R1165 REFUTE ~0.39×**→**R1177 TRAIN** r338 (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:57:59Z | 73490.60477664122 | p4293 **R1160+R1161 REFUTE**→**R1175+R1176 TRAIN** crown (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:51:16Z | 73526.47590989307 | p4292 **R1155+R1140 REFUTE**→**R1173+R1174 TRAIN** (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:42:56Z | 73599.882785395 | p4291 **R1154/57/49/50 REFUTE**→**R1169+70+71+72 TRAIN** (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:31:43Z | 73673.82879972407 | p4290 **R1149+R1150+R1140** n80 relaunch (wrong wait EXP path; no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:24:05Z | 73748.11966023096 | p4289 **R1154** n80 LIVE (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:15:33Z | 73785.56213369875 | p4288 **R1138 REFUTE**→**R1168** SoftCtx HiRank Hiβ UltraLoLR TRAIN (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:09:39Z | 73858.46111010002 | p4287 **R1153 REFUTE**→**R1167** MidLR TRAIN (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:03:33Z | 73891.25386809395 | p4286 **R1145 REFUTE**→**R1166** MidLR TRAIN (no rent; BL-only fbb1135f); burn **~$392.19/h** |
