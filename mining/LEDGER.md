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
| $UTC | 77190.47749891592 | p4237 |
| Lium balance | **$77190.47749891592** | p4237 |
| cumulative mining spend | ~$147,045 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67190** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.18/h** · vs floor $833/h · gap **−$440.82/h** · B300×8=0 · B200×8 stock=0 | p4237 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T01:42:15Z | 77190.47749891592 | p4237 **R1094 REFUTE→R1107 TRAIN** r337 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:36:59Z | 77227.96094820768 | p4236 **R1083 REFUTE→R1106 TRAIN** r337 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:32:48Z | 77265.08638988595 | p4235 **R1081 REFUTE→R1105 TRAIN** r252 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:27:31Z | 77304.26142111408 | p4234 **R1091+R1093 REFUTE→R1103+R1104 TRAIN** crown (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:21:21Z | 77341.24265060782 | p4233 **R1088 REFUTE→R1102 TRAIN** r338 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:14:50Z | 77490.73249001587 | p4232 **R1092 REFUTE→R1101 TRAIN** crown (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:00:40Z | 77532.65637994057 | p4231 **R1087 REFUTE→R1100 TRAIN** r338 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:54:34Z | 77568.49547486355 | p4230 **R1082 REFUTE→R1099 TRAIN** r938 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:47:41Z | 77606.96545503112 | p4229 **R1080 REFUTE→R1098 TRAIN** r926 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:26:03Z | 77759.72796908519 | p4228 **paygo** unstake r252 **88.56α→τ5.06** then **τ5.05→Lium** (lium fund fail→`btcli transfer`); burn **~$392.18/h** |
