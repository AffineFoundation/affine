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
| $UTC | 78639.99309855443 | p4181 |
| Lium balance | **$78639.99309855443** | p4181 |
| cumulative mining spend | ~$143,945 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$68640** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$354.58/h** · vs floor $833/h · gap **−$478.42/h** · B300×8=0 | p4181 |
| miner coldkey free | τ~1249 | kept ≥τ50 |
| miner stake | **0** | p4181 |
| registrations / submissions | **15** / **15** (… **r1032 QUEUED chal-00967**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T18:01:06Z | 78639.99309855443 | p4181 **α→TAO→Lium** r252 88.56α→τ**6.633967769** (lium fund fail→`btcli`→`5FqACMt…zsThe` ext **8887672-0029**) **+$1454** |
| 2026-08-20T17:53:50Z | 77220.46562147482 | p4180 **R1038+R1039 REFUTE→R1049+R1050+R1051 TRAIN** (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:47:01Z | 77254.9071929293 | p4179 **R1025+R1037 REFUTE→R1047+R1048 TRAIN** r337 (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:34:04Z | 77356.82265472399 | p4178 **R1045+R1046 TRAIN** r924 (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:30:16Z | 77425.74534212131 | p4177 **R1032 SUBMITTED** (HF purge LOST ~261GB; burn **~$354.58/h**) |
| 2026-08-20T17:25:30Z | 77425.74534212131 | p4177 **R1032 CROWN_OK→HF push** (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:20:52Z | 77425.74534212131 | p4176 **R1029 REFUTE→R1044 TRAIN** (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:15:15Z | 77495.99119784811 | p4175 **R1025 chall rearm+n80** (no rent; burn **~$354.58/h**) |
| 2026-08-20T17:05:15Z | 77559.67864424792 | p4174 **R1031+R1035+R1036 REFUTE→R1041–43 TRAIN** (no rent; burn **~$354.58/h**) |
| 2026-08-20T15:57:20Z | 78013.00468153591 | p4171 **R339** vera-pin relaunch (fleet 8; burn **~$354.58/h**; no new rent) |
