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
| $UTC | 78876.17344599341 | p4160 |
| Lium balance | **$78876.17344599341** | p4160 |
| cumulative mining spend | ~$142,293 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68876** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 (BL-only B200) | p4160 |
| miner coldkey free | τ~1251 | kept ≥τ50 |
| miner stake | **0** | p4160 |
| registrations / submissions | **14** / **14** (… **r1008** **QUEUED chal-00961**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T13:52:26Z | 78876.17344599341 | p4160 **R1005 REFUTE→R1026 TRAIN** r924 (no rent; burn **~$290.58/h**) |
| 2026-08-20T13:41:35Z | 78946.95394675442 | p4159 **r926 TP4→TP2 + R1025 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T13:26:53Z | 79052.94871544182 | p4158 **TTL+Soft/Dead +24h** all 7 mine-* → removal **2026-08-21T13:26Z**; no rent; burn **~$290.58/h** |
| 2026-08-20T13:23:11Z | 79087.61758226041 | p4157 **α→TAO→Lium** r252 147.6α→τ**8.2526** (`lium fund` fail→`btcli` xfer **8886285-0023**); +**~$1811**; stake=0; burn **~$290.58/h** |
| 2026-08-20T13:18:36Z | 77312.23165756813 | p4156 **R1004+R1011 REFUTE→R1023+R1024 TRAIN** r337; no rent; burn **~$290.58/h** |
| 2026-08-20T13:11:52Z | 77347.28439804418 | p4155 **R1009+R1010 REFUTE→R1021+R1022 TRAIN** crown; no rent; burn **~$290.58/h** |
| 2026-08-20T13:01:06Z | 77415.10014079697 | p4153 **R1006 reap→R1019 TRAIN** crown 1,3; no rent; burn **~$290.58/h** |
| 2026-08-20T12:55:36Z | 77487.18027820435 | p4152 **R998 MERGE→n80** r252 6,7; no rent; burn **~$290.58/h** |
| 2026-08-20T12:49:10Z | 77522.6237510875 | p4151 **R1018 Hiβ TRAIN** r338 4,5; R1008 **chal-00961**; burn **~$290.58/h** |
| 2026-08-20T12:43:56Z | 77558.112247868 | p4150 **R1008 CROWN_OK→SUBMITTED**; R1003→R1016; R1007→R1017; burn **~$290.58/h** |
