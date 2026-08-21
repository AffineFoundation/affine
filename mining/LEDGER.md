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
| $UTC | 72710.26621440689 | p4305 |
| Lium balance | **$72710.26621440689** | p4305 |
| cumulative mining spend | ~$151,531 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$62710** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (10 pods) | **~$424.19/h** · vs floor $833/h · gap **−$408.81/h** · B300/B200/H200×8 stock empty · no new rent | p4305 |
| miner coldkey free | τ~1246.73 | kept ≥τ50 |
| miner stake | **r252 ≈29.52α ≈τ1.69** (<τ5 paygo) | p4235 |
| registrations / submissions | **16** / **16** (… **r1064 LOST chal-00974**) | |

| UTC | Lium USD | event |
|---|---|---|
| 2026-08-21T11:43:58Z | 72710.26621440689 | p4305 **R1175+R1176 REFUTE**→**R1188+R1189 TRAIN** crown (no rent; stock empty); burn **~$424.19/h** |
| 2026-08-21T11:35:48Z | 72788.05287362495 | p4304 **R1158 GRPO TRAIN** + **R1170 REFUTE~0.57×→R1187 TRAIN** (no rent; BL-only fbb1135f); burn **~$424.19/h** |
| 2026-08-21T11:18:06Z | 72902.57954501838 | p4302 **R1167 REFUTE ~−0.15×**→**R1186 TRAIN** r252 (no rent; BL-only fbb1135f; H200×8 avail); burn **~$392.19/h** |
| 2026-08-21T11:06:11Z | 72976.14727272597 | p4301 **R1166 REFUTE ~0.45×**→**R1185 TRAIN** r339 (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T10:56:20Z | 73049.71268173814 | p4300 **R1166** TP2 stall→**TP1 n80 LIVE** r339 (no rent; ls empty post-BL); burn **~$392.19/h** |
| 2026-08-21T10:46:57Z | 73122.78397455945 | p4299 **R1163 REFUTE ~0.18×**→**R1184 TRAIN** r339 (no rent; ls empty post-BL); burn **~$392.19/h** |
| 2026-08-21T10:40:51Z | 73160.85022600633 | p4298 **R1142/44/56 REFUTE**→**R1181+82+83 TRAIN** r340; R1163 n80 LIVE (no rent; ls empty); burn **~$392.19/h** |
| 2026-08-21T10:29:14Z | 73269.40571906604 | p4297 **R1164 REFUTE ~−0.36×**→**R1180 TRAIN** r338 (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T10:22:22Z | 73306.39771039279 | p4296 **R1159+R1162 REFUTE**→**R1178+R1179 TRAIN** (no rent; BL-only fbb1135f); burn **~$392.19/h** |
| 2026-08-21T10:14:48Z | 73379.25539080291 | p4295 **r340 R1142+R1144+R1156 n80 RELAUNCH** TP1/0.85 (no rent; BL-only fbb1135f); burn **~$392.19/h** |
