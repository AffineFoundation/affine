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
| $UTC | 77532.65637994057 | p4231 |
| Lium balance | **$77532.65637994057** | p4231 |
| cumulative mining spend | ~$146,703 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67533** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.18/h** · vs floor $833/h · gap **−$440.82/h** · B300×8=0 · B200×8 stock=0 | p4231 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **0** | p4231 |
| registrations / submissions | **16** / **16** (… **r1064 chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T01:00:40Z | 77532.65637994057 | p4231 **R1087 REFUTE→R1100 TRAIN** r338 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:54:34Z | 77568.49547486355 | p4230 **R1082 REFUTE→R1099 TRAIN** r938 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:47:41Z | 77606.96545503112 | p4229 **R1080 REFUTE→R1098 TRAIN** r926 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:26:03Z | 77759.72796908519 | p4228 **paygo** unstake r252 **88.56α→τ5.06** then **τ5.05→Lium** (lium fund fail→`btcli transfer`); burn **~$392.18/h** |
| 2026-08-21T00:17:41Z | 76687.7937237571 | p4227 r340 king GPU5 **CUDA_HOME fix → :8001 OK** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:09:45Z | 76764.50041797689 | p4226 r340 idle GPUs→**R1096+R1097 TRAIN** + king GPU5 (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-21T00:03:01Z | 76798.78422598669 | p4225 r924 **R1068 idle :8004→R1095 TRAIN** (no rent; B300/B200×8=0); crown SSH timeout; burn **~$392.18/h** |
| 2026-08-20T23:56:33Z | 76840.31469554348 | p4224 r337 **R1064 idle :8003→R1094 TRAIN** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-20T23:50:27Z | 76878.35999003961 | p4223 crown **R1066/67/69 REFUTE→R1091+R1092+R1093 TRAIN** (no rent; B300/B200×8=0); burn **~$392.18/h** |
| 2026-08-20T23:44:33Z | 76955.07838363957 | p4222 **R1076 REFUTE→R1090 TRAIN** r252 (no rent; B300/B200×8=0); burn **~$392.18/h** |
