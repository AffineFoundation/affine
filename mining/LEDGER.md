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
| $UTC | 78436.36776472916 | p4133 |
| Lium balance | **$78436.36776472916** | p4133 |
| cumulative mining spend | ~$141,000 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68436** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 (BL B200 only) | p4133 |
| miner coldkey free | τ~1254 | kept ≥τ50 |
| miner stake | **α29.5/~τ1.66** r252 (under τ5) | p4133 |
| registrations / submissions | **13** / **13** (… **r959** **chal-00957** scoring) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T10:39:42Z | 78436.36776472916 | p4133 **R989 REFUTE**→**R1004 TRAIN** R337 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:31:38Z | 78470.68317155013 | p4132 **R989 MERGE→v4 n80** R337 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:24:17Z | 78542.31394633639 | p4131 **R986 MERGE→n80 waiter** R924 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:18:13Z | 78576.59620472274 | p4130 **R985 REFUTE**→**R1003 TRAIN** R924 4,5 (no rent; 1×B300 only); burn **~$290.58/h** |
| 2026-08-20T10:08:47Z | 78647.16447154088 | p4129 **R988 REFUTE**→**R1002 TRAIN** R252 4,5 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:04:04Z | 78680.31695674355 | p4128 **R984 REFUTE**→**R985 n80**+**R1001 TRAIN** (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T09:56:19Z | 78718.5413279189 | p4127 **4× REFUTE→TRAIN** R997–R1000 + R988 n80 arm (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T09:43:49Z | 78822.82890454572 | p4126 **5× MERGE→v4 n80** R975/R987/R990/R991/R984 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:39:24Z | 78857.77334594833 | p4125 **R973 REFUTE**→**R996 TRAIN** R926 3,4 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T09:33:50Z | 78892.86298275809 | p4124 **R981+R982 REFUTE**→**R994+R995 TRAIN** crown 6,7/4,5 (no rent; B300×8=0); burn **~$290.58/h** |
