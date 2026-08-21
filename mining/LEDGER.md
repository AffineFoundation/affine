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
| $UTC | 69130.20915511684 | p4350 |
| Lium balance | **$69130.20915511684** | p4350 |
| cumulative mining spend | ~$155,074 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$59130** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (12 pods) | **~$480.99/h** · vs floor $833/h · gap **−$352.01/h** | p4350 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T18:38:38Z | 69130.20915511684 | p4350 r340 **R1230+31** + r337 **R1234** n80; **R1227+28 REFUTE**→**R1254+55** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T18:29:11Z | 69220.71706776685 | p4349 crown **R1238+R1239** chall+**v4 n80** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T18:24:03Z | 69263.42759409986 | p4348 r339 **R1227+R1228** chall+**v4 n80** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T18:14:02Z | 69351.62773670067 | p4347 r924 teacher revive + **R1222+23+24** chall+**v4 n80** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T18:04:15Z | 69438.44137351667 | p4346 **R1225+R1226 REFUTE**→**R1252+R1253 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T17:42:21Z | 69617.42012295741 | p4345 r338 **R1225+R1226** chall+**v4 n80 ARMED** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T17:38:21Z | 69660.6239975641 | p4344 r1214 GPUs5–7→**R1249+50+51 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T17:31:29Z | 69703.93398903248 | p4343 r1214 idle→**R1245+46+47+48 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T17:23:58Z | 69793.46582687192 | p4342 r1158 idle→**R1242+43+44 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
| 2026-08-21T17:17:14Z | 69837.72195707068 | p4341 **R1207 REFUTE**→**R1241 TRAIN** (no rent; 8× empty); burn **~$480.99/h** |
