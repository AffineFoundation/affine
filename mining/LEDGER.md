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
| $UTC | 77181.46319228882 | p4217 |
| Lium balance | **$77181.46319228882** | p4217 |
| cumulative mining spend | ~$145,980 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67181** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.18/h** · vs floor $833/h · gap **−$440.82/h** · B300×8=0 · B200×8 stock=0 | p4217 |
| miner coldkey free | τ~1246.7 | kept ≥τ50 |
| miner stake | **r252 ~59α ≈ τ3.36** (<τ5 sweep bar) | p4217 |
| registrations / submissions | **16** / **16** (… **r1064 chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T23:13:22Z | 77181.46319228882 | p4217 **R1078 REFUTE→R1083 TRAIN** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-20T23:07:43Z | 77219.29521916393 | p4216 **R1079 REFUTE→R1082 TRAIN** (no rent); burn **~$392.18/h**; B200×8 stock=1; B300×8=0 |
| 2026-08-20T22:39:38Z | 77447.08707491565 | p4213 **R1060 REFUTE** + **R340 train waiter** (no rent); burn **~$392.18/h**; B300×8=0 |
| 2026-08-20T22:05:55Z | 77750.53055178167 | p4211 **R1060 teacher TP4+n80** (no rent); burn **~$392.18/h**; B300×8=0 |
| 2026-08-20T21:42:59Z | 77861.33248074811 | p4210 **R1060 teacher+n80** + **R340 vera pin**; burn **~$392.18/h**; B300×8=0 |
| 2026-08-20T21:30:46Z | 77929.83713089161 | p4209 **rent B200** `mine-r340`@$37.60/h (gentle-orbit-4a TTL→21:24Z) + R1060 n80 relaunch; burn **~$392.18/h**; B300×8=0 |
| 2026-08-20T21:20:43Z | 77999.62278046655 | p4208 **R1064 Stage-5**: reg + submit reveal **31485871**; **R1062 REFUTE→R1079**; burn **~$354.58/h**; B300×8=0 |
| 2026-08-20T21:07:30Z | 78103.54768824481 | p4207 **α→TAO→Lium**: unstake r252 **59.04α** → `btcli transfer` τ**3.36** Lium · bal **77358→78103** (+~$745) |
| 2026-08-20T21:13:38Z | 78070.45781489636 | p4207 post-burn snapshot (fleet still ~$354.58/h; no rent; B300×8=0) |
| 2026-08-20T21:02:22Z | 77392.16780964343 | p4206 **R1032 LOST** + **R1065 REFUTE→R1077 TRAIN** + R1062 n80 LIVE; B300×8=0 |
| 2026-08-20T20:56:05Z | 77429.71489411636 | p4205 **R1062 ENOSPC→clean+merge retry** r938; B300×8=0 |
| 2026-08-20T20:51:22Z | 77463.66005192383 | p4204 **R1058+R1055 REFUTE→R1075+R1076 TRAIN** r252; B300×8=0 |
