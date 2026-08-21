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
| $UTC | 76735.20851294273 | p4244 |
| Lium balance | **$76735.20851294273** | p4244 |
| cumulative mining spend | ~$147,500 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$66735** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.18/h** · vs floor $833/h · gap **−$440.82/h** · B300×8=0 · B200×8 stock=0 | p4244 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T02:42:19Z | 76735.20851294273 | p4244 **R1098 REFUTE→R1115 TRAIN** r926 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:36:46Z | 76772.35156804496 | p4243 **R1089 REFUTE→R1114 TRAIN** r924 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:32:05Z | 76808.76693606481 | p4242 **R1084 REFUTE→R1113 TRAIN** r924 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:27:50Z | 76849.24800251589 | p4241 **R1095 REFUTE→R1112 TRAIN** r924 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:22:27Z | 76886.84267011733 | p4240 **R1100 REFUTE→R1111 TRAIN** r338 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:17:46Z | 76925.44243119497 | p4239 **R1090 REFUTE→R1110 TRAIN** r252 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:52:54Z | 77113.62750982233 | p4238 **R1085+R1086 REFUTE→R1108+R1109 TRAIN** r339 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:42:15Z | 77190.47749891592 | p4237 **R1094 REFUTE→R1107 TRAIN** r337 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:36:59Z | 77227.96094820768 | p4236 **R1083 REFUTE→R1106 TRAIN** r337 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T01:32:48Z | 77265.08638988595 | p4235 **R1081 REFUTE→R1105 TRAIN** r252 (no rent; B300/B200×8=0); burn **~$392.18/h** |
