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
| $UTC | 78964.04723595499 | p4122 |
| Lium balance | **$78964.04723595499** | p4122 |
| cumulative mining spend | ~$140,928 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68964** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 | p4122 |
| miner coldkey free | τ~1254 | kept ≥τ50 |
| miner stake | **α29.5/~τ1.66** r252 (under τ5) | p4122 |
| registrations / submissions | **13** / **13** (… **r959** **chal-00957** queued) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T09:24:20Z | 78964.04723595499 | p4122 **R980 REFUTE**→**R992 TRAIN** R938 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:19:36Z | 78998.58221574263 | p4121 R973 false key REFUSE→serve HF merge→**v4 n80 LIVE** R926 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:08:48Z | 79069.61823939305 | p4120 crown R981–R983 + R938 R980 **v4 n80 LIVE**; R973 sof REFUSE (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:00:57Z | 79102.00594525364 | p4119 R973 visual/key-layout fail→**p4119b REMERGE sof** R926 (no rent; B300×8=0 BL); burn **~$290.58/h** |
| 2026-08-20T08:40:26Z | 79280.51598051493 | p4118 R973 rematch MERGE→n80 arm R926 (no rent; B300×8=0 BL); burn **~$290.58/h** |
| 2026-08-20T08:34:07Z | 79314.36242093888 | p4117 R978+R979 **REFUTE**→**R990+R991 TRAIN** R338 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:27:44Z | 79350.48888484738 | p4116 R977 **REFUTE**→**R989 TRAIN** + R979 **v4 n80 LIVE** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:22:29Z | 79385.7382217129 | p4115 R978 MERGE idle→**v4 n80 LIVE** R338 6,7 :8002 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:18:04Z | 79411.21362179915 | p4114 R977 MERGE idle→**v4 n80 LIVE** R337 6,7 :8002 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T08:12:03Z | 79455.94744083198 | p4113 R974 **REFUTE**→**R988 TRAIN** R252 4,5 (no rent; B300×8=0); burn **~$290.58/h** |
