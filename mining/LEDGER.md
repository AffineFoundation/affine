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
| $UTC | 72340.29459405222 | p4311 |
| Lium balance | **$72340.29459405222** | p4311 |
| cumulative mining spend | ~$151,901 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$62340** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (11 pods) | **~$456.19/h** · vs floor $833/h · gap **−$376.81/h** · no rent (stock empty) · B300/H200 empty | p4311 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T12:26:15Z | 72340.29459405222 | p4311 **R1174+R1179 REFUTE**→**R1197+R1196 TRAIN** (r938/crown; no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T12:18:42Z | 72422.49359436419 | p4310 **R1177+R1180 REFUTE**→**R1194+R1195 TRAIN** r338 (no rent; BL-only fbb1135f); burn **~$456.19/h** |
| 2026-08-21T12:11:49Z | 72464.16180389075 | p4309 **R1171+R1172 REFUTE**→**R1192+R1193 TRAIN** r337 (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T12:03:27Z | 72545.22105578687 | p4308 fleet TTL +24h (8 pods→22T13:30Z) + Soft/Dead retarget; no rent; burn **~$456.19/h** |
| 2026-08-21T11:58:16Z | 72592.24203144193 | p4307 rented **H200×8** `mine-r1191`/`swift-comet-4d` @$32/h (node `4eb39f3b…`); **R1191 FullFT TRAIN**; burn **~$456.19/h** |
| 2026-08-21T11:49:14Z | 72671.08002065257 | p4306 **R1168 REFUTE~−0.42×**→**R1190 TRAIN** r252 (no rent; BL-only fbb1135f); burn **~$424.19/h** |
| 2026-08-21T11:43:58Z | 72710.26621440689 | p4305 **R1175+R1176 REFUTE**→**R1188+R1189 TRAIN** crown (no rent; stock empty); burn **~$424.19/h** |
| 2026-08-21T11:18:06Z | 72902.57954501838 | p4302 **R1167 REFUTE ~−0.15×**→**R1186 TRAIN** r252 (no rent; BL-only fbb1135f; H200×8 avail); burn **~$392.19/h** |
| 2026-08-21T11:06:11Z | 72976.14727272597 | p4301 **R1166 REFUTE ~0.45×**→**R1185 TRAIN** r339 (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T10:56:20Z | 73049.71268173814 | p4300 **R1166** TP2 stall→**TP1 n80 LIVE** r339 (no rent; ls empty post-BL); burn **~$392.19/h** |
