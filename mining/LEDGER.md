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
| $UTC | 71286.65879855416 | p4323 |
| Lium balance | **$71286.65879855416** | p4323 |
| cumulative mining spend | ~$152,956 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$61287** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (11 pods) | **~$456.19/h** · vs floor $833/h · gap **−$376.81/h** · no rent (8× empty) | p4323 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T14:35:20Z | 71286.65879855416 | p4323 R1192–95 reap→R1210–13 Mega TRAIN (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T14:26:53Z | 71331.32551790909 | p4322 R1187 lean chall relaunch (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T14:20:45Z | 71415.0660734895 | p4321 R1201 REFUTE→R1209 + R1205–08 Mega TRAIN (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T14:11:14Z | 71456.79340261087 | p4320 **R1201** relaunch n80 after teacher OOM@0.90 (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T14:04:03Z | 71536.93233563591 | p4319 **R1181/82/83 REFUTE**→**R1202/03/04 TRAIN** + **R1201 n80** (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T13:34:30Z | 71791.13925530862 | p4318 **r340+r1158+r1191** TTL→22T13:30Z (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T13:30:35Z | 71833.90002176967 | p4317 **R1191 REFUTE~0.02×**→**R1201 TRAIN** (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T13:22:04Z | 71875.24907455561 | p4316 **r340** R1181/82/83 lean chall LOAD (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T13:17:06Z | 71918.43893303456 | p4315 **R1191** pyarrow+T/K TP1→**n80** pid23563 (no rent; stock empty); burn **~$456.19/h** |
| 2026-08-21T12:47:37Z | 72170.79631532395 | p4314 **R1191** merge→local TKC (HF public storage full; no rent; BL-only fbb1135f); burn **~$456.19/h** |
