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
| $UTC | 76394.86134379286 | p4250 |
| Lium balance | **$76394.86134379286** | p4250 |
| cumulative mining spend | ~$147,842 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$66395** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.18/h** · vs floor $833/h · gap **−$440.82/h** · B300×8=0 · B200×8 stock=0 | p4250 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T03:27:46Z | 76394.86134379286 | p4250 **R1107 REFUTE→R1119 TRAIN** r337 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:22:26Z | 76431.02864894563 | p4249 r340 **teacher65536 + R1096/R1097 n80 re-arm** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:12:58Z | 76507.99142783767 | p4248 **R1099 REFUTE→R1118 TRAIN** r938 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T03:03:20Z | 76580.80583604088 | p4247 r340 **Triton-seed+serialize R1096/R1097** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:55:50Z | 76622.00998701864 | p4246 **R1103+R1104 REFUTE→R1116+R1117 TRAIN** crown (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:48:37Z | 76697.1841088578 | p4245 **R1096+R1097 n80 armed** r340 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:42:19Z | 76735.20851294273 | p4244 **R1098 REFUTE→R1115 TRAIN** r926 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:36:46Z | 76772.35156804496 | p4243 **R1089 REFUTE→R1114 TRAIN** r924 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:32:05Z | 76808.76693606481 | p4242 **R1084 REFUTE→R1113 TRAIN** r924 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T02:27:50Z | 76849.24800251589 | p4241 **R1095 REFUTE→R1112 TRAIN** r924 (no rent; B300/B200×8=0); burn **~$392.18/h** |
