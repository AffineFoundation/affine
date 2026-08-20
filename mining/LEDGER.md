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
| $UTC | 78718.5413279189 | p4127 |
| Lium balance | **$78718.5413279189** | p4127 |
| cumulative mining spend | ~$141,000 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68719** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 BL | p4127 |
| miner coldkey free | τ~1254 | kept ≥τ50 |
| miner stake | **α29.5/~τ1.66** r252 (under τ5) | p4127 |
| registrations / submissions | **13** / **13** (… **r959** **chal-00957** queued) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T09:56:19Z | 78718.5413279189 | p4127 **4× REFUTE→TRAIN** R997–R1000 + R988 n80 arm (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T09:43:49Z | 78822.82890454572 | p4126 **5× MERGE→v4 n80** R975/R987/R990/R991/R984 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:39:24Z | 78857.77334594833 | p4125 **R973 REFUTE**→**R996 TRAIN** R926 3,4 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:33:50Z | 78892.86298275809 | p4124 **R981+R982 REFUTE**→**R994+R995 TRAIN** crown 6,7/4,5 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:28:20Z | 78928.74790875013 | p4123 **R983 REFUTE**→**R993 TRAIN** crown 1,3 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:24:20Z | 78964.04723595499 | p4122 **R980 REFUTE**→**R992 TRAIN** R938 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:19:36Z | 78998.58221574263 | p4121 R973 false key REFUSE→serve HF merge→**v4 n80 LIVE** R926 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:08:48Z | 79069.61823939305 | p4120 crown R981–R983 + R938 R980 **v4 n80 LIVE**; R973 sof REFUSE (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:00:57Z | 79102.00594525364 | p4119 R973 visual/key-layout fail→**p4119b REMERGE sof** R926 (no rent; B300×8=0 BL); burn **~$290.58/h** |
| 2026-08-20T08:40:26Z | 79280.51598051493 | p4118 R973 rematch MERGE→n80 arm R926 (no rent; B300×8=0 BL); burn **~$290.58/h** |
