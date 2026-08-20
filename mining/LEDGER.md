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
| $UTC | 77979.92418413499 | p4145 |
| Lium balance | **$77979.92418413499** | p4145 |
| cumulative mining spend | ~$141,415 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67980** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 (BL-only `8f34559f`) | p4145 |
| miner coldkey free | τ~1254 | kept ≥τ50 |
| miner stake | **α29.5/~τ1.66** r252 (under τ5) | p4145 |
| registrations / submissions | **13** / **13** (… **r959** **LOST** chal-00957) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T11:45:00Z | 77979.92418413499 | p4145 **R992 REFUTE**→**R1012 TRAIN** r938 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T11:39:24Z | 78014.34241352418 | p4144 **R996** MERGE idle→**v4 n80** r926 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T11:34:30Z | 78049.60001754439 | p4143 **R997 REFUTE**→**R1011 MidLR TRAIN** r337 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T11:29:09Z | 78085.19445130188 | p4142 **R992** MERGE idle→**v4 n80** r938 (no rent; BL-skip `8f34559f`); burn **~$290.58/h** |
| 2026-08-20T11:24:14Z | 78120.31645445648 | p4141 **R994+R995 REFUTE**→**R1009+R1010 MidLR TRAIN** crown; **R959 LOST**; burn **~$290.58/h** |
| 2026-08-20T11:17:41Z | 78155.10231123568 | p4140 **R999+R1000 REFUTE**→**R1007+R1008 MidLR TRAIN** r338; burn **~$290.58/h** |
| 2026-08-20T11:11:32Z | 78190.33577498296 | p4139 **R993 REFUTE**→**R1006 TRAIN** crown; burn **~$290.58/h** |
| 2026-08-20T11:06:09Z | 78225.50578285327 | p4138 **R999+R1000 MERGE→n80** armed r338; burn **~$290.58/h** |
| 2026-08-20T10:59:49Z | 78296.11566706545 | p4137 **R993+R994+R995 MERGE→n80** armed crown; burn **~$290.58/h** |
| 2026-08-20T10:56:13Z | 78296.11566706545 | p4136 **R986 REFUTE**→**R1005 TRAIN** R924; burn **~$290.58/h** |
