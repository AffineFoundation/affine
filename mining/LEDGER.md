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
| $UTC | 78013.00468153591 | p4171 |
| Lium balance | **$78013.00468153591** | p4171 |
| cumulative mining spend | ~$143,153 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68013** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$354.58/h** · vs floor $833/h · gap **−$478.42/h** · B300×8=0 | p4171 |
| miner coldkey free | τ~1251 | kept ≥τ50 |
| miner stake | **0** | p4171 |
| registrations / submissions | **14** / **14** (… **r1008 LOST** chal-00961) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T15:57:20Z | 78013.00468153591 | p4171 **R339** vera-pin relaunch (fleet 8; burn **~$354.58/h**; no new rent) |
| 2026-08-20T15:51:54Z | 78046.91753114443 | p4170 **R1020 REFUTE→R1040 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:41:27Z | 78115.76692594572 | p4169 **R1027+R1013 REFUTE→R1039 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:31:04Z | 78219.22891395625 | p4168 **R1008 LOST** + **R1028 REFUTE→R1038 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:26:33Z | 78219.22891395625 | p4167 **R1023 REFUTE→R1037 TRAIN** + r926 teacher rearm **R1013 n80** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:06:08Z | 78389.61909739512 | p4166 **R1016+R1024+R1021+R1022 REFUTE→R1033–R1036 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:56:31Z | 78426.52197285187 | p4165 **R1015 REFUTE→R1032 TRAIN** r924 (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:47:49Z | 78493.98998733789 | p4164 **R1014+R1019 REFUTE→R1030+R1031 TRAIN** + R1021/R1022 n80 rearm (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:10:00Z | 78771.9775783544 | p4162 **R1012+R1018 REFUTE→R1029+R1028 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:02:40Z | 78804.35363596232 | p4161 **R1017 REFUTE→R1027 TRAIN** r338 (no rent; burn **~$290.58/h**) |
