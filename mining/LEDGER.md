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
| $UTC | 78389.61909739512 | p4166 |
| Lium balance | **$78389.61909739512** | p4166 |
| cumulative mining spend | ~$142,779 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68390** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 (only 1×B300 stock) | p4166 |
| miner coldkey free | τ~1251 | kept ≥τ50 |
| miner stake | **0** | p4166 |
| registrations / submissions | **14** / **14** (… **r1008** **scoring chal-00961** 1214/1300) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T15:06:08Z | 78389.61909739512 | p4166 **R1016+R1024+R1021+R1022 REFUTE→R1033–R1036 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:56:31Z | 78426.52197285187 | p4165 **R1015 REFUTE→R1032 TRAIN** r924 (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:47:49Z | 78493.98998733789 | p4164 **R1014+R1019 REFUTE→R1030+R1031 TRAIN** + R1021/R1022 n80 rearm (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:10:00Z | 78771.9775783544 | p4162 **R1012+R1018 REFUTE→R1029+R1028 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:02:40Z | 78804.35363596232 | p4161 **R1017 REFUTE→R1027 TRAIN** r338 (no rent; burn **~$290.58/h**) |
| 2026-08-20T13:52:26Z | 78876.17344599341 | p4160 **R1005 REFUTE→R1026 TRAIN** r924 (no rent; burn **~$290.58/h**) |
| 2026-08-20T13:41:35Z | 78946.95394675442 | p4159 **r926 TP4→TP2 + R1025 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T13:26:53Z | 79052.94871544182 | p4158 **TTL+Soft/Dead +24h** all 7 mine-* → removal **2026-08-21T13:26Z**; no rent; burn **~$290.58/h** |
| 2026-08-20T13:23:11Z | 79087.61758226041 | p4157 **α→TAO→Lium** r252 147.6α→τ**8.2526** (`lium fund` fail→`btcli` xfer **8886285-0023**); +**~$1811**; stake=0; burn **~$290.58/h** |
| 2026-08-20T13:18:36Z | 77312.23165756813 | p4156 **R1004+R1011 REFUTE→R1023+R1024 TRAIN** r337; no rent; burn **~$290.58/h** |
