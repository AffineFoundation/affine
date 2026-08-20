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
| $UTC | 77463.66005192383 | p4204 |
| Lium balance | **$77463.66005192383** | p4204 |
| cumulative mining spend | ~$145,021 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67464** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$354.58/h** · vs floor $833/h · gap **−$478.42/h** · B300×8=0 | p4204 |
| miner coldkey free | τ~1249 | kept ≥τ50 |
| miner stake | **~29.5α / τ1.67** (r252) | p4204 |
| registrations / submissions | **15** / **15** (… **r1032 SCORING chal-00967**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T20:51:22Z | 77463.66005192383 | p4204 **R1058+R1055 REFUTE→R1075+R1076 TRAIN** r252 (no rent; burn **~$354.58/h**; B300×8=0 B200×8=0) |
| 2026-08-20T20:42:15Z | 77533.2656349733 | p4203 **R1045+R1054 REFUTE→R1073+R1074** + R1055 chall relaunch (no rent; burn **~$354.58/h**; B300×8=0; bl cosmic-raven-04) |
| 2026-08-20T20:32:45Z | 77601.74605555866 | p4202 **R1061 REFUTE→R1072 TRAIN** r338 (no rent; burn **~$354.58/h**; B300×8=0; bl cosmic-raven-04 only) |
| 2026-08-20T20:26:42Z | 77638.20394121204 | p4201 **R1052 REFUTE→R1071 TRAIN** r339 (no rent; burn **~$354.58/h**; B300×8=0 B200×8=0) |
| 2026-08-20T20:20:42Z | 77671.96871299978 | p4200 **R1052 Triton-reseed n80 LIVE** r339 (no rent; burn **~$354.58/h**; B300×8=0) |
| 2026-08-20T20:10:54Z | 77776.96455726952 | p4199 **R1059+R1053 REFUTE→R1069+R1070 TRAIN** (no rent; burn **~$354.58/h**; B300×8=0) |
| 2026-08-20T20:04:05Z | 77808.4949953878 | p4198 **R1046 REFUTE→R1068 TRAIN** r924 (no rent; burn **~$354.58/h**; skip cosmic-raven-04) |
| 2026-08-20T19:56:49Z | 77847.05559611927 | p4197 **R1057+R1056 REFUTE→R1066+R1067 TRAIN** crown (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:46:51Z | 77915.44390335966 | p4196 **R1050 REFUTE→R1065 TRAIN** r338 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:39:09Z | 77985.20940231917 | p4195 **R1050 Triton-reseed n80 LIVE** r338 (no rent; burn **~$354.58/h**) |
