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
| $UTC | 77929.83713089161 | p4209 |
| Lium balance | **$77929.83713089161** | p4209 |
| cumulative mining spend | ~$145,233 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67930** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.18/h** · vs floor $833/h · gap **−$440.82/h** · B300×8=0 | p4209 |
| miner coldkey free | τ~1246.7 | kept ≥τ50 |
| miner stake | **0** | p4209 |
| registrations / submissions | **16** / **16** (… **r1064 chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T21:30:46Z | 77929.83713089161 | p4209 **rent B200** `mine-r340`@$37.60/h (gentle-orbit-4a TTL→21:24Z) + R1060 n80 relaunch; burn **~$392.18/h**; B300×8=0 |
| 2026-08-20T21:20:43Z | 77999.62278046655 | p4208 **R1064 Stage-5**: reg + submit reveal **31485871**; **R1062 REFUTE→R1079**; burn **~$354.58/h**; B300×8=0 |
| 2026-08-20T21:07:30Z | 78103.54768824481 | p4207 **α→TAO→Lium**: unstake r252 **59.04α** → `btcli transfer` τ**3.36** Lium · bal **77358→78103** (+~$745) |
| 2026-08-20T21:13:38Z | 78070.45781489636 | p4207 post-burn snapshot (fleet still ~$354.58/h; no rent; B300×8=0) |
| 2026-08-20T21:02:22Z | 77392.16780964343 | p4206 **R1032 LOST** + **R1065 REFUTE→R1077 TRAIN** + R1062 n80 LIVE; B300×8=0 |
| 2026-08-20T20:56:05Z | 77429.71489411636 | p4205 **R1062 ENOSPC→clean+merge retry** r938; B300×8=0 |
| 2026-08-20T20:51:22Z | 77463.66005192383 | p4204 **R1058+R1055 REFUTE→R1075+R1076 TRAIN** r252; B300×8=0 |
| 2026-08-20T20:42:15Z | 77533.2656349733 | p4203 **R1045+R1054 REFUTE→R1073+R1074**; B300×8=0 |
| 2026-08-20T20:32:45Z | 77601.74605555866 | p4202 **R1061 REFUTE→R1072 TRAIN** r338; B300×8=0 |
| 2026-08-20T20:26:42Z | 77638.20394121204 | p4201 **R1052 REFUTE→R1071 TRAIN** r339; B300×8=0 |
