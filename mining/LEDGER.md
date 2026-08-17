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
| $UTC | 89984.86832843628 | p3673 |
| Lium balance | **$89984.87** | p3673 |
| cumulative mining spend | ~$113,330 (Δ bal from p526 baseline; includes shared-acct Δ) | $UTC |
| **available for mining** | **~$79985** (balance − $10,000 floor) | $UTC |
| validator burn (never starve) | ~$70/h — eval + bench + teacher2 | 2026-08-14T15:37:00Z |
| miner burn (6 pods) | **~$331.45/h** · vs floor $833/h · gap **−$501.55/h** | $UTC |
| miner coldkey free | τ1261.907 | after unstake+fund (kept ≥τ50) |
| miner stake | **~29.52 α** on r252 (~τ1.68; below τ5 sweep) | 2026-08-17 |
| registrations / submissions | **9** / **9** (… **r252** r33; **r596** chal-00822; **r637** chal-00829) | |

## Recent movements
| UTC | Lium USD | event |
|---|---|---|
| 2026-08-17T14:36:50Z | 89984.86832843628 | p3673 **R634 v4 ARM** lean+wait (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:34:05Z | 90024.43853546199 | p3672 **R658 TRAIN** crown 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:31:17Z | 90024.43853546199 | p3671 **R657 TRAIN** crown 4,5 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:27:35Z | 90065.92614516531 | p3670 **R656 TRAIN** crown 2,3 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:23:49Z | 90106.27616751971 | p3669 **R655 TRAIN** R252 6,7 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:20:25Z | 90106.27616751971 | p3668 **R654 TRAIN** crown 0,1 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:16:47Z | 90146.57365779179 | p3667 **R653 TRAIN** brave 0,1 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:11:31Z | 90186.47632486247 | p3666 **R651 ARMED** lunar (gate R634→R647); **R596 v4 REFUTE**; burn **~$331.45/h** · B300 empty |
| 2026-08-17T14:07:02Z | 90227.26248186323 | p3665 **R596 v4 n80 LIVE** R252 (no rent; B300 empty); burn **~$331.45/h** |
| 2026-08-17T14:03:04Z | 90264.75848219698 | p3664 **fleet v4 affine_pkg sync** (no rent; B300 empty); burn **~$331.45/h** |
