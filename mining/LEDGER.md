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
| $UTC | 73858.46111010002 | p4287 |
| Lium balance | **$73858.46111010002** | p4287 |
| cumulative mining spend | ~$150,382 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$63858** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (9 pods) | **~$392.19/h** · vs floor $833/h · gap **−$440.81/h** · B300×8=0 · B200×8=BL-only | p4287 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T09:09:39Z | 73858.46111010002 | p4287 **R1153 REFUTE**→**R1167** MidLR TRAIN (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T09:03:33Z | 73891.25386809395 | p4286 **R1145 REFUTE**→**R1166** MidLR TRAIN (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T08:55:51Z | 73932.24437318007 | p4285 **R1145** Triton+probe n80 LIVE (no rent; BL-empty); burn **~$392.19/h** |
| 2026-08-21T08:46:03Z | 74042.85244905514 | p4284 **R1143/R1151/R1152 REFUTE**→**R1163/64/65** MidLR + **R1145** Triton-reseed n80 (no rent; BL-empty); burn **~$392.19/h** |
| 2026-08-21T08:36:45Z | 74079.04390478361 | p4283 **R1147 REFUTE**→**R1162** MidLR + r338 **R1151+R1152** n80 relaunch (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T08:26:00Z | 74263.05497162674 | p4282 **R1146+R1148 REFUTE**→**R1160+R1161** TRAIN (no rent; sole B200=BL fbb1135f); burn **~$392.19/h** |
| 2026-08-21T08:10:30Z | 74263.05497162674 | p4281 tore **1/8 GPU** r1158×2 (fbb1135f) + armed node-id waiter; burn **~$392.19/h** |
| 2026-08-21T08:00:48Z | 74332.63946079933 | p4280 **R1139 REFUTE**→**R1159** UltraLoLR + **r1158** bootstrap (no rent; B300/B200×8=0); burn **~$397.79/h** |
| 2026-08-21T07:47:51Z | 74445.4779214407 | p4279 **R1130 REFUTE**→**R1157** + **R1139** n80 + **rent** `mine-r1158` 8×B200 $5.60/h (brave-matrix-2a); burn **~$397.79/h** |
| 2026-08-21T07:32:55Z | 74594.06693887706 | p4278 r340 **R1141 REFUTE**→**R1156** Midβ UltraLoLR TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:26:51Z | 74594.06693887706 | p4277 r924 **R1131 REFUTE**→**R1155** UltraLoLR TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
| 2026-08-21T07:21:28Z | 74666.51591855791 | p4276 r926 **R1130** util0.93 n80 + r924 **R1132 REFUTE**→**R1154** TRAIN (no rent; B300/B200×8=0); burn **~$392.19/h** |
