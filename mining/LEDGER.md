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
| $UTC | 79280.51598051493 | p4118 |
| Lium balance | **$79280.51598051493** | p4118 |
| cumulative mining spend | ~$140,613 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$69281** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 | p4118 |
| miner coldkey free | τ~1254 | kept ≥τ50 |
| miner stake | **α29.5/~τ1.66** r252 (under τ5) | p4118 |
| registrations / submissions | **13** / **13** (… **r959** **chal-00957** queued) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T08:40:26Z | 79280.51598051493 | p4118 R973 rematch MERGE→n80 arm R926 (no rent; B300×8=0 BL); burn **~$290.58/h** |
| 2026-08-20T08:34:07Z | 79314.36242093888 | p4117 R978+R979 **REFUTE**→**R990+R991 TRAIN** R338 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:27:44Z | 79350.48888484738 | p4116 R977 **REFUTE**→**R989 TRAIN** + R979 **v4 n80 LIVE** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:22:29Z | 79385.7382217129 | p4115 R978 MERGE idle→**v4 n80 LIVE** R338 6,7 :8002 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:18:04Z | 79411.21362179915 | p4114 R977 MERGE idle→**v4 n80 LIVE** R337 6,7 :8002 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:12:03Z | 79455.94744083198 | p4113 R974 **REFUTE**→**R988 TRAIN** R252 4,5 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:07:05Z | 79491.02912985809 | p4112 R976 **REFUTE**→**R987 TRAIN** R337 4,5 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:03:20Z | 79523.63494790575 | p4111 R968 **REFUTE**→**R986 TRAIN** + R976+R974 **dual v4 n80 LIVE** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T07:51:52Z | 79631.33711728545 | p4110 R969 **REFUTE**→**R985 TRAIN** + R968 **v4 n80 LIVE** (no rent; B300×8=0 BL); burn **~$290.58/h** |
| 2026-08-20T07:36:27Z | 79701.84131069944 | p4109 α29.5/~τ1.66 r252→**τ1.65**→Lium (`lium fund` Subtensor.transfer fail → `btcli`→`5FqACMt…zsThe`); +~$354; R984 TRAIN; burn **~$290.58/h** |
