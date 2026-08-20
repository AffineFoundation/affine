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
| $UTC | 77915.44390335966 | p4196 |
| Lium balance | **$77915.44390335966** | p4196 |
| cumulative mining spend | ~$144,568 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$67915** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (8 pods) | **~$354.58/h** · vs floor $833/h · gap **−$478.42/h** · B300×8=0 | p4196 |
| miner coldkey free | τ~1249 | kept ≥τ50 |
| miner stake | **~29.5α / τ1.67** (r252) | p4196 |
| registrations / submissions | **15** / **15** (… **r1032 QUEUED chal-00967**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-20T19:46:51Z | 77915.44390335966 | p4196 **R1050 REFUTE→R1065 TRAIN** r338 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:39:09Z | 77985.20940231917 | p4195 **R1050 Triton-reseed n80 LIVE** r338 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:33:01Z | 78019.79528051746 | p4194 **R1047 REFUTE→R1064 TRAIN** r337 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:28:48Z | 78055.74658711975 | p4193 **R1048 REFUTE→R1063 TRAIN** r337 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:19:42Z | 78124.01685769501 | p4192 **R1047+R1048 n80 LIVE** r337 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:15:13Z | 78160.58367926377 | p4191 **R1044 REFUTE→R1062 TRAIN** r938 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:09:05Z | 78193.70446956354 | p4190 **R1049 REFUTE→R1061 TRAIN** r338 (no rent; burn **~$354.58/h**) |
| 2026-08-20T19:03:03Z | 78225.67783080015 | p4189 r926 **TP4→TP2** + **R1060 TRAIN** (no rent; burn **~$354.58/h**) |
| 2026-08-20T18:48:46Z | 78331.50815923551 | p4188 **R1043 REFUTE→R1059 TRAIN** crown (no rent; burn **~$354.58/h**) |
| 2026-08-20T18:40:10Z | 78400.5826534614 | p4187 **R1043 v4 n80 LIVE** crown :8003 (no rent; burn **~$354.58/h**) |
