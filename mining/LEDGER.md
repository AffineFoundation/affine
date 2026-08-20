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
| $UTC | 78296.11566706545 | p4137 |
| Lium balance | **$78296.11566706545** | p4137 |
| cumulative mining spend | ~$141,100 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68296** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 (BL B200 + 1×B300) | p4137 |
| miner coldkey free | τ~1254 | kept ≥τ50 |
| miner stake | **α29.5/~τ1.66** r252 (under τ5) | p4137 |
| registrations / submissions | **13** / **13** (… **r959** **chal-00957** scoring) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T10:59:49Z | 78296.11566706545 | p4137 **R993+R994+R995 MERGE→n80** armed crown (no rent; BL B200 + 1×B300); burn **~$290.58/h** |
| 2026-08-20T10:56:13Z | 78296.11566706545 | p4136 **R986 REFUTE**→**R1005 TRAIN** R924 (no rent; BL B200 + 1×B300); burn **~$290.58/h** |
| 2026-08-20T10:48:13Z | 78365.75079539468 | p4135 **R1001+R1003 MERGE→n80** armed R924 (no rent; BL B200 + 1×B300); burn **~$290.58/h** |
| 2026-08-20T10:44:05Z | 78401.45641619897 | p4134 **R997 MERGE→n80** armed R337 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:39:42Z | 78436.36776472916 | p4133 **R989 REFUTE**→**R1004 TRAIN** R337 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:31:38Z | 78470.68317155013 | p4132 **R989 MERGE→v4 n80** R337 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:24:17Z | 78542.31394633639 | p4131 **R986 MERGE→n80 waiter** R924 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:18:13Z | 78576.59620472274 | p4130 **R985 REFUTE**→**R1003 TRAIN** R924 4,5 (no rent; 1×B300 only); burn **~$290.58/h** |
| 2026-08-20T10:08:47Z | 78647.16447154088 | p4129 **R988 REFUTE**→**R1002 TRAIN** R252 4,5 (no rent; BL B200 only); burn **~$290.58/h** |
| 2026-08-20T10:04:04Z | 78680.31695674355 | p4128 **R984 REFUTE**→**R985 n80**+**R1001 TRAIN** (no rent; BL B200 only); burn **~$290.58/h** |
