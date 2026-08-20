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
| $UTC | 77425.74534212131 | p4176 |
| Lium balance | **$77425.74534212131** | p4176 |
| cumulative mining spend | ~$143,740 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67426** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$354.58/h** · vs floor $833/h · gap **−$478.42/h** · B300×8=0 | p4176 |
| miner coldkey free | τ~1251 | kept ≥τ50 |
| miner stake | **0** | p4176 |
| registrations / submissions | **14** / **14** (… **r1008 LOST** chal-00961) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T17:20:52Z | 77425.74534212131 | p4176 **R1029 REFUTE→R1044 TRAIN** (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:15:15Z | 77495.99119784811 | p4175 **R1025 chall rearm+n80** (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:05:15Z | 77559.67864424792 | p4174 **R1031+R1035+R1036 REFUTE→R1041–43 TRAIN** (no rent; burn **~$354.58/h**) |
| 2026-08-20T15:57:20Z | 78013.00468153591 | p4171 **R339** vera-pin relaunch (fleet 8; burn **~$354.58/h**; no new rent) |
| 2026-08-20T15:51:54Z | 78046.91753114443 | p4170 **R1020 REFUTE→R1040 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:41:27Z | 78115.76692594572 | p4169 **R1027+R1013 REFUTE→R1039 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:31:04Z | 78219.22891395625 | p4168 **R1008 LOST** + **R1028 REFUTE→R1038 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:26:33Z | 78219.22891395625 | p4167 **R1023 REFUTE→R1037 TRAIN** + r926 teacher rearm **R1013 n80** (no rent; burn **~$290.58/h**) |
| 2026-08-20T15:06:08Z | 78389.61909739512 | p4166 **R1016+R1024+R1021+R1022 REFUTE→R1033–R1036 TRAIN** (no rent; burn **~$290.58/h**) |
| 2026-08-20T14:56:31Z | 78426.52197285187 | p4165 **R1015 REFUTE→R1032 TRAIN** r924 (no rent; burn **~$290.58/h**) |
