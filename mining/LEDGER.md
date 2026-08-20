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
| $UTC | 78748.0038939759 | p4090 |
| Lium balance | **$78748.0038939759** | p4090 |
| cumulative mining spend | ~$139,137 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68748** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$329.78/h** · vs floor $833/h · gap **−$503.22/h** · B300×8=0 | p4090 |
| miner coldkey free | τ~1256 | kept ≥τ50 · −τ2.20 reg burn r938 (p4070) |
| miner stake | **~τ3.31** (59α r252) | p4090 · below ~τ5 sweep |
| registrations / submissions | **12** / **12** (… **r938** LOST chal-00949) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T04:59:26Z | 78748.0038939759 | p4090 R926 teacher TP2@0.88→**R944 v4 n80 LIVE** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T04:49:16Z | 78817.30398882508 | p4089 R924 MERGE idle→**R953+R952 v4 n80 LIVE** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T04:13:05Z | 79063.95671809392 | p4087 R944 Triton hang→FORCE seed→**v4 n80 LIVE** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T04:00:11Z | 79169.31253380672 | p4086 R944 thin-pkg fix→**v4 n80 LIVE** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T03:52:25Z | 79203.73966677095 | p4085 R938 **LOST**; R926 king@0.95 READY→R944 chall (no rent; B300×8=0 BL); burn **~$329.78/h** |
| 2026-08-20T03:36:59Z | 79309.67372872595 | p4084 R957 REFUTE→**R967 TRAIN** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T03:28:10Z | 79383.42987896815 | p4083 R957 MERGE idle→**v4 n80 LIVE** (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T03:19:03Z | 79459.52424097626 | p4082 R956+R958 REFUTE→**R965+R966 TRAIN** (no rent; B300×8=0 BL `8f34559f`); burn **~$329.78/h** |
| 2026-08-20T03:10:20Z | 79537.00037931421 | p4081 R955 REFUTE→**R964 TRAIN** + R926 FORCE Triton cold-TK (no rent; B300×8=0); burn **~$329.78/h** |
| 2026-08-20T03:02:40Z | 79572.35041310577 | p4080 R956+R958+R955 MERGE idle→**v4 n80 LIVE** (no rent; B300×8=0 BL); burn **~$329.78/h** |
