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
| $UTC | 70544.91943283178 | p4331 |
| Lium balance | **$70544.91943283178** | p4331 |
| cumulative mining spend | ~$153,656 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$60587** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (12 pods) | **~$480.99/h** · vs floor $833/h · gap **−$352.01/h** | p4331 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T15:59:44Z | 70544.91943283178 | p4331 **R1204** n80 ARMED (no rent; 8× stock empty); burn **~$480.99/h** |
| 2026-08-21T15:52:51Z | 70586.86923286712 | p4330 r924 chall reap→**R1222+23+24 TRAIN** (no rent; 8× stock empty); burn **~$480.99/h** |
| 2026-08-21T15:42:02Z | 70674.78694903315 | p4329 **R1209 REFUTE**→**R1221 TRAIN** + **R1218** single-GPU fix + R1198/99/1200 REFUTE noted (no rent); burn **~$480.99/h** |
| 2026-08-21T15:25:22Z | 70852.8452005539 | p4328 **R1187 REFUTE**→**R1220 TRAIN** + **R1218** affine_pkg flatten relaunch (no rent); burn **~$480.99/h** |
| 2026-08-21T15:17:38Z | 70896.25037611059 | p4327 reap **R1197**→**R1219** SoftCtx LoRank Hiβ Mega HiLR TRAIN (no rent); burn **~$480.99/h** |
| 2026-08-21T15:12:07Z | 70939.75297201177 | p4326 **R1218** BOOT on blank r1214 (no new rent); R1187 n80 28/80; burn **~$480.99/h** |
| 2026-08-21T15:05:38Z | 71073.33034980982 | p4325 crown R1188/89/96→R1215–17 TRAIN + R1187 n80 relaunch (no rent); burn **~$480.99/h** |
| 2026-08-21T14:52:24Z | 71115.61343748964 | p4324 **R1187** n80 ARMED + rent **r1214** H200×8 `$24.80/h`; burn **~$480.99/h** |
| 2026-08-21T14:35:20Z | 71286.65879855416 | p4323 R1192–95 reap→R1210–13 Mega TRAIN; burn **~$456.19/h** |
| 2026-08-21T14:26:53Z | 71331.32551790909 | p4322 R1187 lean chall relaunch; burn **~$456.19/h** |
