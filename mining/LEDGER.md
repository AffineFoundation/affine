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
| $UTC | 69928.2277703054 | p4340 |
| Lium balance | **$69928.2277703054** | p4340 |
| cumulative mining spend | ~$154,273 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$59928** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (12 pods) | **~$480.99/h** · vs floor $833/h · gap **−$352.01/h** | p4340 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T17:09:16Z | 69928.2277703054 | p4340 **R1215+R1216 REFUTE**→**R1239+R1240 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T17:01:22Z | 69963.16526674588 | p4339 **R1217 REFUTE**→**R1238** + r1191 **R1235+36+37 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T16:50:49Z | 70058.00646126471 | p4338 **R1208/10/11 REFUTE**→**R1232+33+34 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T16:44:50Z | 70146.07737916226 | p4337 **R1231 TRAIN** on r340 GPUs6,7 (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T16:40:56Z | 70146.07737916226 | p4336 crown **R1215+16+17** n80 ARMED (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T16:35:00Z | 70234.65365733988 | p4335 **R1202+R1203 REFUTE**→**R1229+R1230 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T16:27:19Z | 70278.66544117211 | p4334 **R1206 REFUTE**→**R1228 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T16:20:34Z | 70323.21103453638 | p4333 **R1206** n80 ARMED (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T16:10:51Z | 70412.59712513175 | p4332 r338→R1225+26 + R1205/04 REFUTE→R1227 + R1202/03 n80 (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T15:59:44Z | 70544.91943283178 | p4331 **R1204** n80 ARMED (no rent; 8× stock empty); burn **~$480.99/h** |
