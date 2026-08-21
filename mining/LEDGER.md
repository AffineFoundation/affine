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
| $UTC | 70939.75297201177 | p4326 |
| Lium balance | **$70939.75297201177** | p4326 |
| cumulative mining spend | ~$153,303 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$60940** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (12 pods) | **~$480.99/h** · vs floor $833/h · gap **−$352.01/h** | p4326 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T15:12:07Z | 70939.75297201177 | p4326 **R1218** BOOT on blank r1214 (no new rent); R1187 n80 28/80; burn **~$480.99/h** |
| 2026-08-21T15:05:38Z | 71073.33034980982 | p4325 crown R1188/89/96→R1215–17 TRAIN + R1187 n80 relaunch (no rent); burn **~$480.99/h** |
| 2026-08-21T14:52:24Z | 71115.61343748964 | p4324 **R1187** n80 ARMED + rent **r1214** H200×8 `$24.80/h`; burn **~$480.99/h** |
| 2026-08-21T14:35:20Z | 71286.65879855416 | p4323 R1192–95 reap→R1210–13 Mega TRAIN; burn **~$456.19/h** |
| 2026-08-21T14:26:53Z | 71331.32551790909 | p4322 R1187 lean chall relaunch; burn **~$456.19/h** |
| 2026-08-21T14:20:45Z | 71415.0660734895 | p4321 R1201 REFUTE→R1209 + R1205–08 Mega TRAIN; burn **~$456.19/h** |
| 2026-08-21T14:11:14Z | 71456.79340261087 | p4320 **R1201** relaunch n80 after teacher OOM@0.90; burn **~$456.19/h** |
| 2026-08-21T14:04:03Z | 71536.93233563591 | p4319 **R1181/82/83 REFUTE**→**R1202/03/04 TRAIN** + **R1201 n80**; burn **~$456.19/h** |
| 2026-08-21T13:34:30Z | 71791.13925530862 | p4318 **r340+r1158+r1191** TTL→22T13:30Z; burn **~$456.19/h** |
| 2026-08-21T13:30:35Z | 71833.90002176967 | p4317 **R1191 REFUTE~0.02×**→**R1201 TRAIN**; burn **~$456.19/h** |
