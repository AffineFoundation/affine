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
| $UTC | 77415.10014079697 | p4153 |
| Lium balance | **$77415.10014079697** | p4153 |
| cumulative mining spend | ~$141,978 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67415** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (7 pods) | **~$290.58/h** · vs floor $833/h · gap **−$542.42/h** · B300×8=0 (only BL B200 stock) | p4153 |
| miner coldkey free | τ~1254 | kept ≥τ50 |
| miner stake | under τ5 | p4153 |
| registrations / submissions | **14** / **14** (… **r1008** **QUEUED chal-00961**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T13:01:06Z | 77415.10014079697 | p4153 **R1006 reap→R1019 TRAIN** crown 1,3; no rent; burn **~$290.58/h** |
| 2026-08-20T12:55:36Z | 77487.18027820435 | p4152 **R998 MERGE→n80** r252 6,7; no rent; burn **~$290.58/h** |
| 2026-08-20T12:49:10Z | 77522.6237510875 | p4151 **R1018 Hiβ TRAIN** r338 4,5; R1008 **chal-00961**; burn **~$290.58/h** |
| 2026-08-20T12:43:56Z | 77558.112247868 | p4150 **R1008 CROWN_OK→SUBMITTED**; R1003→R1016; R1007→R1017; burn **~$290.58/h** |
| 2026-08-20T12:19:47Z | 77733.09083896516 | p4149 **R1001 REFUTE**→**R1015 MidLR TRAIN** r924 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T12:05:36Z | 77835.7925770603 | p4148 **R996+R1002 REFUTE**→**R1013+R1014 MidLR TRAIN** (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T11:56:37Z | 77873.26088961706 | p4147 rented+rm `fbb1135f` lemon (~$0.26 spent); burn **~$290.58/h**; B300×8=0 |
| 2026-08-20T11:49:19Z | 77944.38212357037 | p4146 **R1002 MERGE→n80** armed r252 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T11:45:00Z | 77979.92418413499 | p4145 **R992 REFUTE**→**R1012 TRAIN** r938 (no rent; B300×8=0); burn **~$290.58/h** |
| 2026-08-20T11:39:24Z | 78014.34241352418 | p4144 **R996** MERGE idle→**v4 n80** r926 (no rent; B300×8=0); burn **~$290.58/h** |
